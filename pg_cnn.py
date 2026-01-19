import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _extract_board(observation):
    if isinstance(observation, dict):
        if "board" in observation:
            return np.asarray(observation["board"])
        if "observation" in observation:
            return np.asarray(observation["observation"])
        for value in observation.values():
            if hasattr(value, "shape"):
                return np.asarray(value)
        raise ValueError("No array-like observation found in dict.")
    return np.asarray(observation)


def preprocess_observation(observation):
    board = _extract_board(observation)
    if board.ndim == 2:
        board = board[None, :, :]
    elif board.ndim == 3:
        if board.shape[-1] in (1, 2, 3, 4) and board.shape[0] not in (1, 2, 3, 4):
            board = np.transpose(board, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported observation shape: {board.shape}")
    board = board.astype(np.float32)
    max_val = float(board.max()) if board.size else 1.0
    if max_val > 1.0:
        board = board / max_val
    return torch.from_numpy(board)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        channels, height, width = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


def select_action(policy, obs_tensor, device, greedy=False):
    logits = policy(obs_tensor.unsqueeze(0).to(device))
    if greedy:
        action = torch.argmax(logits, dim=-1)
        return int(action.item()), None
    dist = Categorical(logits=logits)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action).squeeze(0)


def compute_returns(rewards, gamma):
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)
