#!/usr/bin/env python3
"""
Local training script for Kaggriculture PPO agent with enhanced reward function.
Implements relative profit maximization with constraint enforcement.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import random
import numpy as np
import torch

from encoder import ObservationEncoder
from executor import MacroActionExecutor
from values import relative_potential
from config import Config
from agent import PPOAgent, League, FrozenOpponent
from environment import KaggricultureEnv


def run_episode(env, encoder, executor, agent, opponent, self_play,
                max_steps=720, core_mask=None):
    obs = env.reset()
    done = False
    step = 0

    h0 = agent.new_hidden()
    h1 = agent.new_hidden() if self_play else None

    ep0 = "seat0"
    agent.begin_episode(ep0)
    if self_play:
        ep1 = "seat1"
        agent.begin_episode(ep1)

    while not done and step < max_steps:
        step += 1
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


def create_env(executor, reward_config=None, shaped_weight=None):
    from environment import KaggricultureEnv
    
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
            },
            "shaped_weight": shaped_weight if shaped_weight is not None else 1.0,
        }
    
    return KaggricultureEnv(executor=executor, reward_config=reward_config, shaping_gamma=1.0)


def train(
    config=None,
    num_iterations=100,
    episodes_per_iter=4,
    p_self_play=0.5,
    checkpoint_path="checkpoints/checkpoint.pt",
    resume=False,
):
    if config is None:
        config = Config()

    device = config.device
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    executor = MacroActionExecutor()
    encoder = ObservationEncoder()
    
    reward_config = {
        "invalid_penalty": config.reward_invalid_penalty,
        "do_nothing_penalty": config.reward_do_nothing_penalty,
        "do_nothing_threshold": config.reward_do_nothing_threshold,
        "terminal_win_bonus": config.reward_terminal_win_bonus,
        "terminal_loss_penalty": config.reward_terminal_loss_penalty,
        "terminal_tie_bonus": config.reward_terminal_tie_bonus,
        "effectiveness_bonus": {
            "harvest": config.reward_effectiveness_harvest,
            "sell": config.reward_effectiveness_sell,
            "plant": config.reward_effectiveness_plant,
        },
        "shaped_weight": config.reward_shaped_weight,
    }

    env = create_env(
        executor=executor,
        reward_config=reward_config,
        shaped_weight=config.reward_shaped_weight,
    )

    obs_dim = encoder.output_dim
    global_dim = 2  # money, shed_value
    hidden_dim = config.hidden_dim
    num_actions = executor.num_macros
    


    agent = PPOAgent(
        obs_dim=obs_dim,
        global_dim=global_dim,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        lr=config.lr,
        gamma=config.gamma,
        lam=config.lam,
        clip_range=config.clip_range,
        ppo_epochs=config.ppo_epochs,
        chunk_len=config.chunk_len,
        entropy_coef=config.entropy_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        device=device,
    )

    league = League(max_size=config.league_max_size)
    history = []

    start_iter = 0

    if checkpoint_path and os.path.exists(checkpoint_path) and resume:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        agent.reward_rms.mean = checkpoint["reward_rms_mean"]
        agent.reward_rms.var = checkpoint["reward_rms_var"]
        agent.reward_rms.count = checkpoint["reward_rms_count"]
        if "league" in checkpoint:
            league.snapshots = checkpoint["league"]
        if "history" in checkpoint:
            history = checkpoint["history"]
            start_iter = len(history)
        print(f"  Resumed from iteration {start_iter}")

    if config.checkpoint_every > 0:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    pbar = range(start_iter, num_iterations)

    try:
        for it in pbar:
            iter_moneys = []
            
            for ep in range(episodes_per_iter):
                if random.random() < p_self_play or len(league) == 0:
                    m0, m1 = run_episode(
                        env, encoder, executor, agent,
                        opponent=None,
                        self_play=True,
                        max_steps=config.max_steps,
                    )
                else:
                    snap = league.sample()
                    opponent = FrozenOpponent(
                        obs_dim, hidden_dim, num_actions, snap, device=device
                    )
                    m0, m1 = run_episode(
                        env, encoder, executor, agent,
                        opponent=opponent,
                        self_play=False,
                        max_steps=config.max_steps,
                    )
                iter_moneys.append(m0)

                if config.debug_reward_check and agent.episodes:
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

            history.append({
                "iter": len(history),
                "mean_money": float(np.mean(iter_moneys)),
                "p_self_play": p_self_play,
            })

            print(f"Iter {it+1}/{num_iterations}: mean_money={np.mean(iter_moneys):.0f}, "
                  f"league_size={len(league)}, p_self={p_self_play:.2f}")
            
            # Visualize every 10 iterations
            if (it + 1) % 10 == 0:
                print("  [Visualization] Running evaluation episode...")
                try:
                    viz_frames = visualize_episode(encoder, executor, agent, max_steps=100)
                    if viz_frames:
                        print(f"  [Visualization] Generated {len(viz_frames)} frames")
                except Exception as e:
                    print(f"  [Visualization] Error: {e}")

            if (it + 1) % config.league_add_every == 0:
                league.add(agent.actor_state_dict())

            if checkpoint_path and (it + 1) % config.checkpoint_every == 0:
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
                }, checkpoint_path)

    except KeyboardInterrupt:
        print("Interrupted -- returning current agent/league/history so far.")

    if checkpoint_path:
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
        }, checkpoint_path)

    return agent, league, history


def visualize_episode(encoder, executor, agent, max_steps=720, show_html=True):
    """Run an episode and return frames for visualization."""
    from kaggle_environments import make
    
    env = make("kaggriculture", debug=True)
    env.reset()
    done = False
    step = 0
    frames = []
    
    h0 = agent.new_hidden()
    ep0 = "viz"
    agent.begin_episode(ep0)
    
    while not done and step < max_steps:
        step += 1
        obs = env.steps[-1][0].observation
        obs0 = encoder.encode(obs)
        mask0 = executor.legal_macros(obs)
        
        action0, _, h0 = agent.act(obs0, h0, mask=mask0)
        result = env.step({0: action0, 1: action0})
        done = env.done
        
        if show_html:
            frames.append(env.render(mode='html'))
    
    agent.end_episode(ep0)
    
    return frames


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Kaggriculture agent locally")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iter", type=int, default=4)
    parser.add_argument("--p-self-play", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint.pt")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    config = Config(seed=args.seed, device="cpu" if args.no_cuda else "cuda")
    
    train(
        config=config,
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        p_self_play=args.p_self_play,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
