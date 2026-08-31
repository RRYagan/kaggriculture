import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_actions):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_actions)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.head(out), hidden


class MLPCritic(nn.Module):
    def __init__(self, global_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)
