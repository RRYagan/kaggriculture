import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np

from executor import MacroActionExecutor
from values import relative_potential
from config import Config


def run_episode(env, encoder, executor, agent, opponent, self_play,
                max_steps=720, core_mask=None):
    obs = env.reset()
    done = False
    step = 0
    day = 1

    h0 = agent.new_hidden()
    h1 = agent.new_hidden() if self_play else None

    ep0 = "seat0"
    agent.begin_episode(ep0)
    if self_play:
        ep1 = "seat1"
        agent.begin_episode(ep1)

    while not done and step < max_steps:
        step += 1
        day = obs[0].get("day", 1)
        obs0 = encoder.encode(obs[0])
        obs1 = encoder.encode(obs[1])
        global0 = encoder.encode_global(obs, perspective=0)
        mask0 = executor.legal_macros(obs[0])
        mask1 = executor.legal_macros(obs[1])

        if core_mask is not None:
            mask0 = mask0 & core_mask
            mask1 = mask1 & core_mask

        action0, logp0, h0 = agent.act(obs0, h0, mask=mask0)
        if self_play:
            action1, logp1, h1 = agent.act(obs1, h1, mask=mask1)
        else:
            action1, h1 = opponent.act(obs1, h1, mask=mask1)

        next_obs, rewards, dones, _ = env.step({0: action0, 1: action1})
        done = dones[0]

        agent.store(ep0, obs0, action0, rewards[0], global0, done, logp0, mask=mask0)
        if self_play:
            global1 = encoder.encode_global(obs, perspective=1)
            agent.store(ep1, obs1, action1, rewards[1], global1, done, logp1, mask=mask1)

        obs = next_obs

    agent.end_episode(ep0)
    if self_play:
        agent.end_episode(ep1)

    return obs[0]["farms"][0]["money"], obs[1]["farms"][1]["money"]


def save_checkpoint(agent, league, history, path):
    import torch
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "actor_opt": agent.actor_optimizer.state_dict(),
        "critic_opt": agent.critic_optimizer.state_dict(),
        "reward_rms_mean": agent.reward_rms.mean,
        "reward_rms_var": agent.reward_rms.var,
        "reward_rms_count": agent.reward_rms.count,
        "league": league.snapshots,
        "history": history,
    }, path)


def load_checkpoint(agent, league, path):
    import torch
    try:
        ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=agent.device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor_optimizer.load_state_dict(ckpt["actor_opt"])
    agent.critic_optimizer.load_state_dict(ckpt["critic_opt"])
    agent.reward_rms.mean = ckpt["reward_rms_mean"]
    agent.reward_rms.var = ckpt["reward_rms_var"]
    agent.reward_rms.count = ckpt["reward_rms_count"]
    league.snapshots = ckpt["league"]
    return ckpt.get("history", [])


def train_league_selfplay(
    agent,
    league,
    encoder,
    executor,
    env,
    num_iterations=50,
    episodes_per_iter=4,
    p_self_play=0.5,
    p_self_play_schedule=None,
    league_add_every=10,
    max_steps=720,
    checkpoint_every=10,
    checkpoint_path=None,
    history=None,
    fixed_opponent=None,
    core_actions=None,
    debug_reward_check=False,
    reward_config=None,
):
    if reward_config is None:
        reward_config = Config(
            reward_invalid_penalty=-20.0,
            reward_do_nothing_penalty=-10.0,
            reward_do_nothing_threshold=2,
            reward_terminal_win_bonus=2000.0,
            reward_terminal_loss_penalty=-1000.0,
            reward_terminal_tie_bonus=500.0,
            reward_shaped_weight=1.0,
        )
    history = history if history is not None else []
    obs_dim, hidden_dim = encoder.output_dim, agent.actor.lstm.hidden_size
    num_actions = executor.num_macros

    if len(league) == 0:
        league.add(agent.actor_state_dict())

    core_mask = None
    if core_actions is not None:
        core_mask = np.zeros(num_actions, dtype=bool)
        core_mask[core_actions] = True

    try:
        from tqdm import trange
    except ImportError:
        trange = range

    pbar = trange(num_iterations, desc="training")
    try:
        for it in pbar:
            current_p_self = p_self_play_schedule(it) if p_self_play_schedule else p_self_play
            iter_moneys = []
            for ep in range(episodes_per_iter):
                if fixed_opponent is not None:
                    m0, m1 = run_episode(
                        env, encoder, executor, agent,
                        opponent=fixed_opponent,
                        self_play=False,
                        max_steps=max_steps,
                        core_mask=core_mask,
                    )
                elif random.random() < current_p_self or len(league) == 0:
                    m0, m1 = run_episode(
                        env, encoder, executor, agent,
                        opponent=None,
                        self_play=True,
                        max_steps=max_steps,
                        core_mask=core_mask,
                    )
                else:
                    snap = league.sample()
                    opponent = FrozenOpponent(
                        obs_dim, hidden_dim, num_actions, snap, device=agent.device
                    )
                    m0, m1 = run_episode(
                        env, encoder, executor, agent,
                        opponent=opponent,
                        self_play=False,
                        max_steps=max_steps,
                        core_mask=core_mask,
                    )
                iter_moneys.append(m0)

                if debug_reward_check and agent.episodes:
                    last_ep = agent.episodes[-1]
                    raw_rew = np.array(last_ep["rew"])
                    total_shaped = float(raw_rew.sum())
                    initial_money = env.initial_money[0]
                    profit = m0 - initial_money
                    print(
                        f"  [check] ep {ep}: shaped_total={total_shaped:.2f}, "
                        f"final_money={m0:.2f}, start_money={initial_money:.2f}, "
                        f"profit={profit:.2f}, abs_diff={abs(total_shaped - profit):.4f}"
                    )

            agent.update()

            history.append(
                {
                    "iter": len(history),
                    "mean_money": float(np.mean(iter_moneys)),
                    "p_self_play": current_p_self,
                }
            )
            pbar.set_postfix(
                mean_money=f"{np.mean(iter_moneys):.0f}",
                league=len(league),
                p_self=f"{current_p_self:.2f}",
            )

            if (it + 1) % league_add_every == 0:
                league.add(agent.actor_state_dict())

            if checkpoint_path and (it + 1) % checkpoint_every == 0:
                save_checkpoint(agent, league, history, checkpoint_path)

    except KeyboardInterrupt:
        print("Interrupted -- returning current agent/league/history so far.")

    if checkpoint_path:
        save_checkpoint(agent, league, history, checkpoint_path)

    return agent, league, history


def create_env(executor, reward_config=None, shaped_weight=None):
    from src.environment import KaggricultureEnv
    
    if reward_config is None:
        reward_config = {
            "invalid_penalty": -20.0,
            "do_nothing_penalty": -10.0,
            "do_nothing_threshold": 2,
            "terminal_win_bonus": 2000.0,
            "terminal_loss_penalty": -1000.0,
            "terminal_tie_bonus": 500.0,
            "effectiveness_bonus": {
                "harvest": 1.0,
                "sell": 0.5,
                "plant": 0.3,
                "water": 0.2,
                "feed": 0.3,
            },
            "shaped_weight": shaped_weight if shaped_weight is not None else 1.0,
        }
    
    return KaggricultureEnv(executor=executor, reward_config=reward_config, shaping_gamma=1.0)
