import copy
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import trange

from src.encoder import ObservationEncoder
from src.executor import MacroActionExecutor
from src.config import Config
from src.environment import KaggricultureEnv

TYPE_NAMES = [
    "do_nothing",
    "water_all",
    "harvest_ready",
    "hire_hand",
    "buy_land",
    "feed_animals",
    "care_animals",
    "fertilize_ready",
    "sell",
    "plant",
    "buy_place_animal",
]

NEEDS_ARG = {"sell", "plant", "buy_place_animal"}

SELL_LEVELS = [0.25, 0.5, 0.75, 1.0]
CROP_LIST = ["wheat", "corn", "barley", "oats", "potatoes"]
ANIMAL_LIST = ["chickens", "cows"]


class LSTMPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_types: int, n_args: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.type_logits = torch.nn.Linear(hidden_dim, n_types)
        self.arg_head = torch.nn.Linear(hidden_dim, n_args)

    def forward_features(self, x: torch.Tensor, hidden):
        features, new_hidden = self.lstm(x, hidden)
        return features, new_hidden

    def type_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.type_logits(features)

    def arg_logits(self, features: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = features.shape
        type_emb = type_id.view(batch_size * seq_len).long()
        arg_logits = self.arg_head(features)
        return arg_logits


class RunningStd:
    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PPOAgent:
    def __init__(self, obs_dim: int, global_dim: int, n_types: int, n_args: int,
                 hidden_dim: int = 128, lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_range: float = 0.2, ppo_epochs: int = 4, chunk_len: int = 16,
                 entropy_coef: float = 0.01, vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 reward_std_floor: float = 1e-4, debug_prints: bool = False,
                 device: str = "cpu"):
        self.device = device
        self.actor = LSTMPolicy(obs_dim, hidden_dim, n_types, n_args).to(device)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(global_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.ppo_epochs = ppo_epochs
        self.chunk_len = chunk_len
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.reward_rms = RunningStd()
        self.reward_std_floor = reward_std_floor
        self.debug_prints = debug_prints
        self.episodes: List[Dict] = []
        self.current_episode: Dict = {
            "obs": [], "tid": [], "aid": [], "rew": [], "global_obs": [],
            "done": [], "logp": [], "active": [], "type_mask": [], "arg_mask": []
        }

    def new_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def begin_episode(self, episode_id: str):
        self.current_episode = {
            "obs": [], "tid": [], "aid": [], "rew": [], "global_obs": [],
            "done": [], "logp": [], "active": [], "type_mask": [], "arg_mask": []
        }

    def store(self, obs: np.ndarray, tid: int, aid: int, rew: float, global_obs: np.ndarray,
              done: bool, logp: float, active: bool, type_mask: np.ndarray, arg_mask: np.ndarray):
        self.current_episode["obs"].append(obs)
        self.current_episode["tid"].append(tid)
        self.current_episode["aid"].append(aid)
        self.current_episode["rew"].append(rew)
        self.current_episode["global_obs"].append(global_obs)
        self.current_episode["done"].append(done)
        self.current_episode["logp"].append(logp)
        self.current_episode["active"].append(active)
        self.current_episode["type_mask"].append(type_mask)
        self.current_episode["arg_mask"].append(arg_mask)

    def end_episode(self, episode_id: str):
        if len(self.current_episode["obs"]) == 0:
            return
        self.episodes.append(copy.deepcopy(self.current_episode))
        self.current_episode = {
            "obs": [], "tid": [], "aid": [], "rew": [], "global_obs": [],
            "done": [], "logp": [], "active": [], "type_mask": [], "arg_mask": []
        }

    def act(self, obs: np.ndarray, hidden, type_mask: np.ndarray, arg_mask_by_type: Dict) -> Tuple:
        self.actor.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
            features, new_hidden = self.actor.forward_features(x, hidden)
            type_logits = self.actor.type_logits(features).squeeze(0).squeeze(0)
            tm = torch.as_tensor(type_mask, dtype=torch.bool, device=self.device)
            type_logits = type_logits.masked_fill(~tm, float("-inf"))
            type_dist = Categorical(logits=type_logits)
            type_id = type_dist.sample()
            tid = type_id.item()
            logp = type_dist.log_prob(type_id).item()
            entropy = type_dist.entropy().item()
            arg_active = int(tid in NEEDS_ARG)
            if arg_active:
                arg_logits = self.actor.arg_logits(features, type_id).squeeze(0).squeeze(0)
                am = torch.as_tensor(arg_mask_by_type.get(tid, []), dtype=torch.bool, device=self.device)
                arg_logits = arg_logits.masked_fill(~am, float("-inf"))
                arg_dist = Categorical(logits=arg_logits)
                aid = arg_dist.sample().item()
                logp += arg_dist.log_prob(torch.tensor(aid, device=self.device)).item()
                entropy += arg_dist.entropy().item()
            else:
                aid = 0
        return tid, aid, logp, arg_active, arg_mask_by_type.get(tid, []), new_hidden

    def act_greedy(self, obs: np.ndarray, hidden, type_mask: np.ndarray, arg_mask_by_type: Dict) -> Tuple:
        self.actor.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
            features, new_hidden = self.actor.forward_features(x, hidden)
            type_logits = self.actor.type_logits(features).squeeze(0).squeeze(0)
            tm = torch.as_tensor(type_mask, dtype=torch.bool, device=self.device)
            type_logits = type_logits.masked_fill(~tm, float("-inf"))
            type_id = type_logits.argmax().item()
            arg_active = int(type_id in NEEDS_ARG)
            if arg_active:
                arg_logits = self.actor.arg_logits(features, torch.tensor([[type_id]], device=self.device)).squeeze(0).squeeze(0)
                am = torch.as_tensor(arg_mask_by_type.get(type_id, []), dtype=torch.bool, device=self.device)
                arg_logits = arg_logits.masked_fill(~am, float("-inf"))
                aid = arg_logits.argmax().item()
            else:
                aid = 0
        return type_id, aid, new_hidden

    def update(self):
        if not self.episodes:
            return
        for ep in self.episodes:
            rews = np.array(ep["rew"])
            self.reward_rms.update(rews)
            rews = (rews - self.reward_rms.mean) / max(np.std(rews), self.reward_rms.std_floor)
            returns = []
            cum_return = 0
            for r in reversed(rews):
                cum_return = r + self.gamma * cum_return
                returns.insert(0, cum_return)
            returns = np.array(returns)
            ep["ret"] = returns
        for _ in range(self.ppo_epochs):
            for ep in self.episodes:
                obs_chunk = torch.as_tensor(np.array(ep["obs"]), dtype=torch.float32, device=self.device).unsqueeze(0)
                ret_chunk = torch.as_tensor(ep["ret"], dtype=torch.float32, device=self.device).unsqueeze(0)
                old_logp_chunk = torch.as_tensor(ep["logp"], dtype=torch.float32, device=self.device).unsqueeze(0)
                type_id_chunk = torch.as_tensor(ep["tid"], dtype=torch.long, device=self.device).unsqueeze(0)
                arg_id_chunk = torch.as_tensor(ep["aid"], dtype=torch.long, device=self.device).unsqueeze(0)
                active_chunk = torch.as_tensor(ep["active"], dtype=torch.float32, device=self.device).unsqueeze(0)
                global_chunk = torch.as_tensor(np.array(ep["global_obs"]), dtype=torch.float32, device=self.device).unsqueeze(0)
                type_mask_chunk = torch.as_tensor(np.array(ep["type_mask"]), dtype=torch.bool, device=self.device).unsqueeze(0)
                arg_mask_chunk = torch.as_tensor(np.array(ep["arg_mask"]), dtype=torch.bool, device=self.device).unsqueeze(0)
                adv_chunk = ret_chunk - self.critic(global_chunk).squeeze(-1)
                features, hidden = self.actor.forward_features(obs_chunk, (torch.zeros(1, 1, self.hidden_dim, device=self.device),
                                                                           torch.zeros(1, 1, self.hidden_dim, device=self.device)))
                hidden = tuple(h.detach() for h in hidden)
                type_logits = self.actor.type_logits(features)
                type_logits = type_logits.masked_fill(~type_mask_chunk, float("-inf"))
                type_dist = Categorical(logits=type_logits)
                type_logp = type_dist.log_prob(type_id_chunk)
                type_entropy = type_dist.entropy()
                arg_logits = self.actor.arg_logits(features, type_id_chunk)
                arg_logits = arg_logits.masked_fill(~arg_mask_chunk, float("-inf"))
                arg_dist = Categorical(logits=arg_logits)
                arg_logp = arg_dist.log_prob(arg_id_chunk)
                arg_entropy = arg_dist.entropy()
                new_logp = type_logp + active_chunk * arg_logp
                entropy = (type_entropy + active_chunk * arg_entropy).mean()
                ratio = torch.exp(new_logp - old_logp_chunk)
                surr1 = ratio * adv_chunk
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_chunk
                actor_loss = -torch.min(surr1, surr2).mean()
                values_chunk = self.critic(global_chunk)
                critic_loss = F.mse_loss(values_chunk, ret_chunk)
                if self.debug_prints:
                    print(f"  ratio mean={ratio.mean().item():.3f} max={ratio.max().item():.3f} "
                          f"adv mean={adv_chunk.mean().item():.3f} entropy={entropy.item():.3f} "
                          f"actor_loss={actor_loss.item():.3f} critic_loss={critic_loss.item():.3f}")
                total_loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def actor_state_dict(self):
        return copy.deepcopy(self.actor.state_dict())

    def load_actor_state_dict(self, sd):
        self.actor.load_state_dict(sd)


class FrozenOpponent:
    def __init__(self, obs_dim: int, hidden_dim: int, n_types: int, n_args: int,
                 state_dict, device: str = "cpu"):
        self.actor = LSTMPolicy(obs_dim, hidden_dim, n_types, n_args).to(device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        self.device = device
        self.n_args = n_args

    def act(self, obs, hidden, type_mask=None, arg_mask_by_type=None):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
            features, new_hidden = self.actor.forward_features(x, hidden)
            type_logits = self.actor.type_logits(features).squeeze(0).squeeze(0)
            if type_mask is not None:
                tm = torch.as_tensor(type_mask, dtype=torch.bool, device=self.device)
                type_logits = type_logits.masked_fill(~tm, float("-inf"))
            type_dist = Categorical(logits=type_logits)
            type_id = type_dist.sample()
            tid = type_id.item()
            arg_id_val = 0
            if arg_mask_by_type is not None and tid in arg_mask_by_type:
                type_id_b = type_id.view(1, 1)
                arg_logits = self.actor.arg_logits(features, type_id_b).squeeze(0).squeeze(0)
                am = torch.as_tensor(arg_mask_by_type[tid], dtype=torch.bool, device=self.device)
                arg_logits = arg_logits.masked_fill(~am, float("-inf"))
                arg_id_val = torch.distributions.Categorical(logits=arg_logits).sample().item()
        return tid, arg_id_val, new_hidden


class ScriptedOpponent:
    def __init__(self, type_id: int, arg_id: int = 0):
        self.type_id = type_id
        self.arg_id = arg_id

    def act(self, obs, hidden, type_mask=None, arg_mask_by_type=None):
        return self.type_id, self.arg_id, hidden


class League:
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.snapshots = []

    def add(self, actor_state_dict):
        snap = copy.deepcopy(actor_state_dict)
        if len(self.snapshots) >= self.max_size:
            self.snapshots.pop(0)
        self.snapshots.append(snap)

    def sample(self):
        return random.choice(self.snapshots) if self.snapshots else None

    def __len__(self):
        return len(self.snapshots)


def build_arg_mask_by_type(executor: MacroActionExecutor, obs, type_mask: np.ndarray) -> Dict:
    return {
        t: executor.legal_args(obs, t)
        for t in NEEDS_ARG
        if type_mask[t]
    }


def run_episode(env, encoder: ObservationEncoder, executor: MacroActionExecutor,
                agent: PPOAgent, opponent, self_play: bool,
                max_steps: int = 720, core_type_mask: np.ndarray = None):
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
        type_mask0 = executor.legal_types(obs[0])
        type_mask1 = executor.legal_types(obs[1])
        if core_type_mask is not None:
            type_mask0 = type_mask0 & core_type_mask
            type_mask1 = type_mask1 & core_type_mask
            if not type_mask0.any():
                type_mask0[0] = True
            if not type_mask1.any():
                type_mask1[0] = True
        arg_mask_by_type0 = build_arg_mask_by_type(executor, obs[0], type_mask0)
        arg_mask_by_type1 = build_arg_mask_by_type(executor, obs[1], type_mask1)
        tid0, aid0, logp0, active0, argmask0, h0 = agent.act(obs0, h0, type_mask0, arg_mask_by_type0)
        if self_play:
            tid1, aid1, logp1, active1, argmask1, h1 = agent.act(obs1, h1, type_mask1, arg_mask_by_type1)
        else:
            tid1, aid1, h1 = opponent.act(obs1, h1, type_mask1, arg_mask_by_type1)
        action0 = executor.execute(obs[0], tid0, aid0)
        action1 = executor.execute(obs[1], tid1, aid1)
        next_obs, rewards, dones, _ = env.step({0: action0, 1: action1})
        done = dones[0]
        agent.store(ep0, obs0, tid0, aid0, rewards[0], global0, done, logp0,
                    active0, type_mask0, argmask0)
        if self_play:
            global1 = encoder.encode_global(obs, perspective=1)
            agent.store(ep1, obs1, tid1, aid1, rewards[1], global1, done, logp1,
                        active1, type_mask1, argmask1)
        obs = next_obs
    agent.end_episode(ep0)
    if self_play:
        agent.end_episode(ep1)
    return obs[0]["farms"][0]["money"], obs[1]["farms"][1]["money"]


def save_checkpoint(agent: PPOAgent, league: League, history: List, path: str):
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


def load_checkpoint(agent: PPOAgent, league: League, path: str) -> List:
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
    agent: PPOAgent,
    league: League,
    encoder: ObservationEncoder,
    executor: MacroActionExecutor,
    env: KaggricultureEnv,
    num_iterations: int = 50,
    episodes_per_iter: int = 4,
    p_self_play: float = 0.5,
    p_self_play_schedule = None,
    league_add_every: int = 10,
    max_steps: int = 720,
    checkpoint_every: int = 10,
    checkpoint_path: str = None,
    history: List = None,
    fixed_opponent = None,
    core_actions: List = None,
    debug_reward_check: bool = False,
):
    history = history if history is not None else []
    obs_dim, hidden_dim = encoder.output_dim, agent.actor.lstm.hidden_size
    n_types, n_args = executor.n_types, executor.n_args
    if len(league) == 0:
        league.add(agent.actor_state_dict())
    core_type_mask = None
    if core_actions is not None:
        core_type_mask = np.zeros(n_types, dtype=bool)
        core_type_mask[core_actions] = True
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
                        core_type_mask=core_type_mask,
                    )
                elif random.random() < current_p_self or len(league) == 0:
                    m0, m1 = run_episode(
                        env, encoder, executor, agent,
                        opponent=None,
                        self_play=True,
                        max_steps=max_steps,
                        core_type_mask=core_type_mask,
                    )
                else:
                    snap = league.sample()
                    opponent = FrozenOpponent(
                        obs_dim, hidden_dim, n_types, n_args, snap, device=agent.device
                    )
                    m0, m1 = run_episode(
                        env, encoder, executor, agent,
                        opponent=opponent,
                        self_play=False,
                        max_steps=max_steps,
                        core_type_mask=core_type_mask,
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


def do_nothing_action(executor: MacroActionExecutor):
    return TYPE_NAMES.index("do_nothing"), 0


def evaluate(agent: PPOAgent, encoder: ObservationEncoder, executor: MacroActionExecutor,
             env: KaggricultureEnv, opponent_state_dict = None, num_episodes: int = 5,
             max_steps: int = 720, hidden_dim: int = 128, deterministic: bool = True,
             verbose: bool = True, core_actions: List = None):
    obs_dim = encoder.output_dim
    n_types, n_args = executor.n_types, executor.n_args
    opponent = None
    if opponent_state_dict is not None:
        opponent = FrozenOpponent(obs_dim, hidden_dim, n_types, n_args, opponent_state_dict)
    core_type_mask = None
    if core_actions is not None:
        core_type_mask = np.zeros(n_types, dtype=bool)
        core_type_mask[core_actions] = True
    results = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        h0 = agent.new_hidden()
        h1 = agent.new_hidden() if opponent is not None else None
        while not done and step < max_steps:
            step += 1
            obs0 = encoder.encode(obs[0])
            obs1 = encoder.encode(obs[1])
            type_mask0 = executor.legal_types(obs[0])
            type_mask1 = executor.legal_types(obs[1])
            if core_type_mask is not None:
                type_mask0 = type_mask0 & core_type_mask
                type_mask1 = type_mask1 & core_type_mask
                if not type_mask0.any():
                    type_mask0[0] = True
                if not type_mask1.any():
                    type_mask1[0] = True
            arg_mask_by_type0 = build_arg_mask_by_type(executor, obs[0], type_mask0)
            arg_mask_by_type1 = build_arg_mask_by_type(executor, obs[1], type_mask1)
            if deterministic:
                tid0, aid0, h0 = agent.act_greedy(obs0, h0, type_mask0, arg_mask_by_type0)
            else:
                tid0, aid0, _, _, _, h0 = agent.act(obs0, h0, type_mask0, arg_mask_by_type0)
            if opponent is not None:
                tid1, aid1, h1 = opponent.act(obs1, h1, type_mask1, arg_mask_by_type1)
            else:
                tid1, aid1 = do_nothing_action(executor)
            action0 = executor.execute(obs[0], tid0, aid0)
            action1 = executor.execute(obs[1], tid1, aid1)
            next_obs, rewards, dones, _ = env.step({0: action0, 1: action1})
            done = dones[0]
            obs = next_obs
        money0 = obs[0]["farms"][0]["money"]
        money1 = obs[1]["farms"][1]["money"]
        won = money0 > money1
        results.append((money0, money1, won))
        if verbose:
            print(f"[eval ep {ep+1}/{num_episodes}] agent_money={money0:.0f} "
                  f"opponent_money={money1:.0f} agent_won={won}")
    agent_moneys = [r[0] for r in results]
    win_rate = sum(r[2] for r in results) / len(results)
    print(f"\nMean agent final money: {np.mean(agent_moneys):.1f} "
          f"(+/- {np.std(agent_moneys):.1f})  |  win rate: {win_rate:.0%}")
    return results


def plot_training_curve(history: List):
    import matplotlib.pyplot as plt
    iters = [h["iter"] for h in history]
    money = [h["mean_money"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(iters, money, label="mean money (self-play episodes)")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("mean money")
    if "p_self_play" in history[0]:
        ax2 = ax1.twinx()
        ax2.plot(iters, [h["p_self_play"] for h in history], color="gray", alpha=0.4, label="p_self_play")
        ax2.set_ylabel("p_self_play")
    fig.tight_layout()
    plt.show()


def debug_action_distribution(agent: PPOAgent, encoder: ObservationEncoder,
                              executor: MacroActionExecutor, env: KaggricultureEnv,
                              num_episodes: int = 1, max_steps: int = 720,
                              core_actions: List = None):
    from collections import Counter
    type_counts = Counter()
    type_arg_counts = Counter()
    n_types = executor.n_types
    core_type_mask = None
    if core_actions is not None:
        core_type_mask = np.zeros(n_types, dtype=bool)
        core_type_mask[core_actions] = True
    obs = env.reset()
    done = False
    step = 0
    h0 = agent.new_hidden()
    while not done and step < max_steps:
        step += 1
        obs0 = encoder.encode(obs[0])
        obs1 = encoder.encode(obs[1])
        type_mask0 = executor.legal_types(obs[0])
        if core_type_mask is not None:
            type_mask0 = type_mask0 & core_type_mask
            if not type_mask0.any():
                type_mask0[0] = True
        arg_mask_by_type0 = build_arg_mask_by_type(executor, obs[0], type_mask0)
        tid0, aid0, h0 = agent.act_greedy(obs0, h0, type_mask0, arg_mask_by_type0)
        type_counts[tid0] += 1
        if tid0 in NEEDS_ARG:
            type_arg_counts[(tid0, aid0)] += 1
        tid1, aid1 = do_nothing_action(executor)
        action0 = executor.execute(obs[0], tid0, aid0)
        action1 = executor.execute(obs[1], tid1, aid1)
        next_obs, rewards, dones, _ = env.step({0: action0, 1: action1})
        done = dones[0]
        obs = next_obs
    print("Type distribution (greedy) over one episode:")
    for tid, count in sorted(type_counts.items()):
        print(f"  Type {tid:2d} ({TYPE_NAMES[tid]:20s}): {count}")
    if type_arg_counts:
        print("\nArgument breakdown for types that use one:")
        arg_lists = {"sell": SELL_LEVELS, "plant": CROP_LIST, "buy_place_animal": ANIMAL_LIST}
        for (tid, aid), count in sorted(type_arg_counts.items()):
            name = TYPE_NAMES[tid]
            arg_name = arg_lists.get(name, [str(i) for i in range(executor.n_args)])[aid]
            print(f"  {name:18s} arg={arg_name:12s}: {count}")
