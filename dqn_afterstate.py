import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# Feature indices for the Dellacherie feature vector.
FEATURE_HOLES = 0
FEATURE_BUMPINESS = 1
FEATURE_AGG_HEIGHT = 2
FEATURE_LINES = 3
FEATURE_DIM = 4

# Heuristic: the env uses a padded board; skip the top padding rows.
DEFAULT_TOP_PADDING = 2


def extract_board(observation):
    """Pull the raw board array from a Gymnasium observation dict or array."""
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


def extract_lines_cleared(info):
    """Try to read lines-cleared from the info dict; fall back to 0."""
    if not info:
        return 0
    for key in ("lines_cleared", "cleared_lines", "lines", "line_clears", "rows_cleared"):
        if key in info:
            try:
                return int(info[key])
            except (TypeError, ValueError):
                return 0
    return 0


def _playable_columns(board):
    """Detect playable columns by excluding padding columns with top-row == 1."""
    top_row = board[0]
    mask = top_row != 1
    if not mask.any():
        # If padding is not present, treat every column as playable.
        mask = np.ones_like(top_row, dtype=bool)
    return mask


def extract_features(boards, lines_cleared=None, top_padding=DEFAULT_TOP_PADDING):
    """
    Vectorized Dellacherie feature extraction.

    Features per board:
      - holes: empty cells with a filled cell above
      - bumpiness: sum of abs diff between adjacent column heights
      - aggregate height: sum of column heights
      - lines cleared: supplied externally from the last move
    """
    boards = np.asarray(boards)
    if boards.ndim == 2:
        boards = boards[None, :, :]
    if boards.ndim != 3:
        raise ValueError(f"Unsupported board shape: {boards.shape}")

    # Crop to the playable area (remove padding rows/columns).
    playable_cols = _playable_columns(boards[0])
    row_start = top_padding if boards.shape[1] > top_padding else 0
    playable = boards[:, row_start:, playable_cols]

    # Binary occupancy: any non-zero cell counts as filled.
    occupied = playable != 0
    rows = playable.shape[1]

    # Column heights: distance from top to highest filled cell.
    any_filled = occupied.any(axis=1)
    first_filled = np.argmax(occupied, axis=1)
    heights = np.where(any_filled, rows - first_filled, 0)

    # Aggregate height is the sum of column heights.
    aggregate_height = heights.sum(axis=1)

    # Holes: empty cells that have a filled cell above them in the same column.
    filled_above = np.cumsum(occupied, axis=1) > 0
    holes = np.logical_and(~occupied, filled_above).sum(axis=(1, 2))

    # Bumpiness: sum of absolute adjacent column height differences.
    bumpiness = np.abs(np.diff(heights, axis=1)).sum(axis=1)

    # Lines cleared is supplied from the environment info, not the board.
    batch = boards.shape[0]
    if lines_cleared is None:
        lines = np.zeros(batch, dtype=np.float32)
    else:
        lines = np.asarray(lines_cleared, dtype=np.float32)
        if lines.ndim == 0:
            lines = np.full(batch, lines, dtype=np.float32)

    features = np.stack([holes, bumpiness, aggregate_height, lines], axis=1).astype(
        np.float32
    )
    return features


def build_action_sequences(max_shift=10):
    """Enumerate macro-actions: rotations + left/right shifts + hard drop."""
    sequences = []
    rotations = [[], [3], [3, 3], [3, 3, 3]]
    for rot in rotations:
        for k in range(max_shift + 1):
            sequences.append(rot + [0] * k + [5])
            sequences.append(rot + [1] * k + [5])
    unique = []
    seen = set()
    for seq in sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique.append(seq)
    return unique


def simulate_sequence(env, sequence):
    """Play a macro-action on a copied env to get the resulting after-state."""
    new_env = copy.deepcopy(env)
    total_reward = 0.0
    terminated = False
    truncated = False
    obs = None
    info = {}
    for action in sequence:
        obs, reward, terminated, truncated, info = new_env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    if obs is None:
        raise RuntimeError("Sequence produced no observation.")
    return obs, total_reward, terminated or truncated, info


def run_sequence(env, sequence, render_fn=None):
    """Play a macro-action on the real env and return the resulting state."""
    total_reward = 0.0
    terminated = False
    truncated = False
    obs = None
    info = {}
    for action in sequence:
        if render_fn is not None:
            render_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    if obs is None:
        raise RuntimeError("Sequence produced no observation.")
    return obs, total_reward, terminated or truncated, info


def enumerate_after_states(env, sequences):
    """Return candidate after-states with precomputed feature vectors."""
    boards = []
    infos = []
    dones = []
    for seq in sequences:
        obs, _, done, info = simulate_sequence(env, seq)
        boards.append(extract_board(obs).copy())
        infos.append(info)
        dones.append(done)

    boards_batch = np.stack(boards, axis=0)
    lines_cleared = np.array(
        [extract_lines_cleared(info) for info in infos], dtype=np.float32
    )
    features = extract_features(boards_batch, lines_cleared)

    candidates = []
    for idx, seq in enumerate(sequences):
        candidates.append(
            {
                "sequence": seq,
                "features": features[idx],
                "lines_cleared": int(lines_cleared[idx]),
                "done": dones[idx],
            }
        )
    return candidates


class AfterStateQNetwork(nn.Module):
    """Simple MLP that maps Dellacherie features to a scalar value."""

    def __init__(self, num_features=FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """FIFO replay buffer storing after-state features and next candidates."""

    def __init__(self, capacity):
        self._buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self._buffer)

    def push(self, state_features, reward, done, next_features):
        self._buffer.append((state_features, reward, done, next_features))

    def sample(self, batch_size):
        batch = random.sample(self._buffer, batch_size)
        states, rewards, dones, next_states = zip(*batch)
        return states, rewards, dones, next_states
