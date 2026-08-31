import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningStd:
    """Welford's online algorithm -- tracks a running std to rescale rewards
    without needing full history in memory. Scale only, don't subtract the
    mean, so the potential-based shaping guarantee isn't disturbed by an
    added per-step constant drift."""
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = len(x)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = float(M2 / tot_count)
        self.count = float(tot_count)

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-8


class PPOAgent:
    def __init__(self, obs_dim, global_dim, num_actions, hidden_dim=128, lr=1e-4,
                 gamma=1.0, lam=0.95, clip_range=0.2, ppo_epochs=4, chunk_len=32,
                 entropy_coef=0.2, vf_coef=0.5, max_grad_norm=0.5, device="cuda",
                 shaped_weight=None):
        self.obs_dim, self.global_dim, self.num_actions = obs_dim, global_dim, num_actions
        self.gamma, self.lam, self.clip_range = gamma, lam, clip_range
        self.ppo_epochs, self.chunk_len = ppo_epochs, chunk_len
        self.entropy_coef, self.vf_coef, self.max_grad_norm = entropy_coef, vf_coef, max_grad_norm
        self.device = device

        from networks import LSTMPolicy, MLPCritic
        self.actor = LSTMPolicy(obs_dim, hidden_dim, num_actions).to(device)
        self.critic = MLPCritic(global_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.episodes = []
        self._open = {}
        self.reward_rms = RunningStd()
        
        self.shaped_weight = shaped_weight if shaped_weight is not None else 1.0

    def new_hidden(self):
        return None

    def _masked_logits(self, logits, mask):
        if mask is None:
            return logits
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        return logits.masked_fill(~mask_t, float("-inf"))

    def act(self, obs, hidden, mask=None):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
            logits, new_hidden = self.actor(x, hidden)
            logits = self._masked_logits(logits.squeeze(0).squeeze(0), mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), new_hidden

    def act_greedy(self, obs, hidden, mask=None):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
            logits, new_hidden = self.actor(x, hidden)
            logits = self._masked_logits(logits.squeeze(0).squeeze(0), mask)
            action = logits.argmax().item()
        return action, new_hidden

    def begin_episode(self, episode_id):
        self._open[episode_id] = {"obs": [], "act": [], "rew": [], "global": [],
                                   "done": [], "logp": [], "mask": []}

    def store(self, episode_id, obs, action, reward, global_state, done, log_prob, mask=None):
        ep = self._open[episode_id]
        ep["obs"].append(obs); ep["act"].append(action); ep["rew"].append(reward)
        ep["global"].append(global_state); ep["done"].append(done); ep["logp"].append(log_prob)
        ep["mask"].append(mask if mask is not None else np.ones(self.num_actions, dtype=bool))

    def end_episode(self, episode_id):
        ep = self._open.pop(episode_id)
        T = len(ep["obs"])
        if T == 0:
            return
        with torch.no_grad():
            global_t = torch.as_tensor(np.array(ep["global"], dtype=np.float32), device=self.device).unsqueeze(0)
            values = self.critic(global_t).squeeze(0).cpu().numpy()

        raw_rewards = np.array(ep["rew"], dtype=np.float32)
        self.reward_rms.update(raw_rewards)
        rewards = raw_rewards

        dones = np.array(ep["done"], dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)
        next_value, next_adv = 0.0, 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * next_adv
            next_value, next_adv = values[t], advantages[t]
        returns = advantages + values
        adv_mean, adv_std = advantages.mean(), advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        ep["adv"], ep["ret"] = advantages, returns
        self.episodes.append(ep)

    def update(self):
        if not self.episodes:
            return
        for _ in range(self.ppo_epochs):
            random.shuffle(self.episodes)
            for ep in self.episodes:
                self._update_on_episode(ep)
        self.episodes.clear()

    def _update_on_episode(self, ep):
        T = len(ep["obs"])
        obs = torch.as_tensor(np.array(ep["obs"], dtype=np.float32), device=self.device)
        acts = torch.as_tensor(np.array(ep["act"], dtype=np.int64), device=self.device)
        old_logp = torch.as_tensor(np.array(ep["logp"], dtype=np.float32), device=self.device)
        adv = torch.as_tensor(ep["adv"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(ep["ret"], dtype=torch.float32, device=self.device)
        global_state = torch.as_tensor(np.array(ep["global"], dtype=np.float32), device=self.device)
        masks = torch.as_tensor(np.array(ep["mask"], dtype=bool), device=self.device)

        hidden = None
        for start in range(0, T, self.chunk_len):
            end = min(start + self.chunk_len, T)
            obs_chunk = obs[start:end].unsqueeze(0)
            act_chunk = acts[start:end].unsqueeze(0)
            old_logp_chunk = old_logp[start:end].unsqueeze(0)
            adv_chunk = adv[start:end].unsqueeze(0)
            ret_chunk = ret[start:end]
            global_chunk = global_state[start:end]
            mask_chunk = masks[start:end].unsqueeze(0)

            logits, hidden = self.actor(obs_chunk, hidden)
            logits = logits.masked_fill(~mask_chunk, float("-inf"))
            hidden = tuple(h.detach() for h in hidden)

            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_chunk)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_chunk)
            surr1 = ratio * adv_chunk
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_chunk
            actor_loss = -torch.min(surr1, surr2).mean()

            values_chunk = self.critic(global_chunk)
            critic_loss = F.mse_loss(values_chunk, ret_chunk)

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
    def __init__(self, obs_dim, hidden_dim, num_actions, state_dict, device="cuda"):
        from networks import LSTMPolicy
        self.actor = LSTMPolicy(obs_dim, hidden_dim, num_actions).to(device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        self.device = device

    def act(self, obs, hidden, mask=None):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
            logits, new_hidden = self.actor(x, hidden)
            logits = logits.squeeze(0).squeeze(0)
            if mask is not None:
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(~mask_t, float("-inf"))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action.item(), new_hidden


class ScriptedOpponent:
    def __init__(self, macro_id):
        self.macro_id = macro_id

    def act(self, obs, hidden, mask=None):
        return self.macro_id, hidden


class League:
    def __init__(self, max_size=20):
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
