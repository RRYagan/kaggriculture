import dataclasses
from typing import Optional, List


@dataclasses.dataclass
class Config:
    seed: int = 0
    board_size: int = 10
    device: str = "cpu"

    # network / PPO
    hidden_dim: int = 128
    lr: float = 1e-4
    gamma: float = 1.0
    lam: float = 0.95
    clip_range: float = 0.2
    ppo_epochs: int = 4
    chunk_len: int = 32
    entropy_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_std_floor: float = 1.0
    debug_prints: bool = False

    # env / training
    num_iterations: int = 100
    episodes_per_iter: int = 4
    p_self_play: float = 0.5
    league_add_every: int = 20
    league_max_size: int = 20
    max_steps: int = 720
    checkpoint_every: int = 10
    # core_actions: Optional[List[int]] = None
    core_actions = [0, 1, 2, 3, 9]
    debug_reward_check: bool = False

    # reward function configuration
    reward_invalid_penalty: float = -20.0
    reward_do_nothing_penalty: float = -10.0
    reward_do_nothing_threshold: int = 2
    reward_terminal_win_bonus: float = 2000.0
    reward_terminal_loss_penalty: float = -1000.0
    reward_terminal_tie_bonus: float = 500.0
    reward_shaped_weight: float = 1.0
    reward_effectiveness_harvest: float = 1.0
    reward_effectiveness_sell: float = 0.5
    reward_effectiveness_plant: float = 0.3
    reward_effectiveness_water: float = 0.2
    reward_effectiveness_feed: float = 0.3

    def seed_everything(self):
        import random, numpy as np, torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
