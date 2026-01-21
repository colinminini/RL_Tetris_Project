import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tetris_gymnasium.envs import Tetris

from DQN_scripts.dqn_afterstate import (
    AfterStateQNetwork,
    FEATURE_BUMPINESS,
    FEATURE_DIM,
    FEATURE_HOLES,
    FEATURE_LINES,
    ReplayBuffer,
    build_action_sequences,
    enumerate_after_states,
    extract_board,
    extract_features,
    extract_lines_cleared,
    run_sequence,
)

# Dense line-clear shaping: 1 -> 1, 2 -> 3, 3 -> 6, 4 -> 12.
# Keys are line counts (int), values are shaped rewards (float).
LINE_CLEAR_REWARDS = {1: 1.0, 2: 3.0, 3: 6.0, 4: 12.0}


def make_env(render):
    """Create the Gymnasium Tetris environment."""
    # render=True enables human-mode rendering with scaled window.
    if render:
        return gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    return gym.make("tetris_gymnasium/Tetris")


def save_checkpoint(path, policy, target, optimizer, episode):
    """Persist training state for later evaluation."""
    # payload stores training metadata + model/optimizer weights.
    payload = {
        "episode": episode,
        "feature_dim": FEATURE_DIM,
        "model_state": policy.state_dict(),
        "target_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    save_dir = os.path.dirname(path)
    if save_dir:
        # Ensure the directory exists before saving.
        os.makedirs(save_dir, exist_ok=True)
    # torch.save serializes the Python dict to a .pt file on disk.
    torch.save(payload, path)


def parse_args():
    # CLI for training hyperparameters and logging.
    parser = argparse.ArgumentParser(
        description="Train a DQN after-state agent with Dellacherie features."
    )
    # episodes: total training episodes; gamma: discount; lr: Adam learning rate.
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    # seed: RNG seed; device: "auto" or explicit torch device string.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    # max-steps: optional cap on steps per episode.
    parser.add_argument("--max-steps", type=int, default=None)
    # replay buffer and training batch sizes.
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--start-training", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=500)
    # epsilon-greedy schedule settings.
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=int, default=20000)
    # max-shift: number of left/right shifts to consider per macro-action.
    parser.add_argument("--max-shift", type=int, default=10)
    # log/save intervals measured in episodes.
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--save-path", type=str, default="tetris_code/checkpoints/dqn_afterstate.pt"
    )
    parser.add_argument("--render", action="store_true")
    # max-grad-norm: gradient clipping threshold (None disables).
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    return parser.parse_args()


def epsilon_by_step(step, start, end, decay):
    """Exponential epsilon schedule."""
    # step: int global step; start/end: float eps bounds; decay: float time constant.
    if decay <= 0:
        return end
    mix = np.exp(-step / decay)
    # mix in [0, 1]; epsilon decays from start to end.
    return end + (start - end) * mix


def compute_candidate_rewards(prev_features, candidate_features):
    """
    Vectorized reward shaping:
      + line reward
      - holes created
      - bumpiness increase
      + survival bonus
    """
    # prev_features: np.ndarray shape (FEATURE_DIM,)
    # candidate_features: np.ndarray shape (num_candidates, FEATURE_DIM)
    lines = candidate_features[:, FEATURE_LINES]
    # lines: np.ndarray shape (num_candidates,), entries are line counts per candidate
    line_rewards = (
        (lines == 1) * LINE_CLEAR_REWARDS[1]
        + (lines == 2) * LINE_CLEAR_REWARDS[2]
        + (lines == 3) * LINE_CLEAR_REWARDS[3]
        + (lines == 4) * LINE_CLEAR_REWARDS[4]
    )
    # line_rewards: np.ndarray shape (num_candidates,), dense reward per candidate

    # Penalize only *new* holes and *increased* bumpiness.
    holes_delta = np.maximum(0.0, candidate_features[:, FEATURE_HOLES] - prev_features[FEATURE_HOLES])
    # holes_delta: np.ndarray shape (num_candidates,), clamp to new holes only
    bump_delta = np.maximum(
        0.0, candidate_features[:, FEATURE_BUMPINESS] - prev_features[FEATURE_BUMPINESS]
    )
    # bump_delta: np.ndarray shape (num_candidates,), clamp to increased bumpiness only

    # Small survival bonus per placed piece.
    survival_bonus = 0.1
    # shaped: np.ndarray shape (num_candidates,), float32 rewards for each candidate
    shaped = line_rewards - 2.0 * holes_delta - 0.5 * bump_delta + survival_bonus
    return shaped.astype(np.float32)


def select_action(policy, candidate_features, candidate_rewards, device, epsilon, gamma):
    """Epsilon-greedy action selection using reward + gamma * V(after-state)."""
    if candidate_features.shape[0] == 0:
        raise RuntimeError("No candidate after-states available.")
    if random.random() < epsilon:
        return random.randrange(candidate_features.shape[0])

    # Batch all candidates for a single GPU/MPS forward pass.
    # candidate_features: np.ndarray shape (num_candidates, FEATURE_DIM)
    # candidate_rewards: np.ndarray shape (num_candidates,)
    # torch.from_numpy shares memory with NumPy (CPU) and preserves dtype.
    features_tensor = torch.from_numpy(candidate_features).to(device)
    rewards_tensor = torch.from_numpy(candidate_rewards).to(device)
    # torch.no_grad disables gradient tracking for inference-time forward pass.
    with torch.no_grad():
        # policy forward: (num_candidates, FEATURE_DIM) -> (num_candidates, 1)
        # squeeze(-1) removes the trailing singleton dimension -> (num_candidates,)
        values = policy(features_tensor).squeeze(-1)
    # scores: torch.Tensor shape (num_candidates,), reward + gamma * value
    scores = rewards_tensor + gamma * values
    # torch.argmax returns index of the max score (long tensor) -> convert to int
    return int(torch.argmax(scores).item())


def compute_batch_loss(policy, target, batch, gamma, device):
    """DDQN loss: policy selects next state, target evaluates it."""
    states, rewards, dones, next_states = batch
    # states: tuple of np.ndarray shape (FEATURE_DIM,)
    # rewards: tuple of float, dones: tuple of bool
    # next_states: tuple of np.ndarray shape (num_next, FEATURE_DIM) or empty

    # Current after-state values from the policy network.
    # np.stack -> (batch_size, FEATURE_DIM); float32 for torch
    state_tensor = torch.from_numpy(np.stack(states).astype(np.float32)).to(device)
    # policy forward: (batch_size, FEATURE_DIM) -> (batch_size, 1)
    q_pred = policy(state_tensor).squeeze(-1)
    # q_pred: torch.Tensor shape (batch_size,), predicted V(after-state)

    targets = []
    for reward, done, next_after_features in zip(rewards, dones, next_states):
        # reward: scalar float, done: bool
        # next_after_features: np.ndarray shape (num_next, FEATURE_DIM) or empty
        if done or getattr(next_after_features, "size", 0) == 0:
            targets.append(float(reward))
            continue

        # torch.from_numpy -> CPU tensor; to(device) moves to GPU/MPS/CPU
        next_tensor = torch.from_numpy(next_after_features.astype(np.float32)).to(device)
        # torch.no_grad disables gradient tracking for target/policy evaluation.
        with torch.no_grad():
            # DDQN: policy picks the best next after-state.
            # policy output: (num_next, 1) -> squeeze -> (num_next,)
            next_q_policy = policy(next_tensor).squeeze(-1)
            # torch.argmax returns the index of max Q-value
            best_idx = int(torch.argmax(next_q_policy).item())
            # Target network evaluates that choice.
            # target output: (num_next, 1) -> squeeze -> (num_next,)
            next_q_target = target(next_tensor).squeeze(-1)[best_idx].item()
        targets.append(float(reward) + gamma * next_q_target)

    # target_tensor: torch.Tensor shape (batch_size,), float32 targets
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    # F.mse_loss computes mean squared error between q_pred and target_tensor
    return F.mse_loss(q_pred, target_tensor)


def main():
    args = parse_args()
    if args.device == "auto":
        # Prefer MPS for Apple Silicon, then CUDA, then CPU.
        if torch.backends.mps.is_available():
            # torch.device("mps") routes tensors/ops to Apple Metal backend.
            device = torch.device("mps")
        elif torch.cuda.is_available():
            # torch.device("cuda") routes tensors/ops to NVIDIA GPU.
            device = torch.device("cuda")
        else:
            # torch.device("cpu") keeps tensors/ops on host CPU.
            device = torch.device("cpu")
    else:
        # Use the explicit device string provided (e.g., "cpu", "cuda:0").
        device = torch.device(args.device)

    if args.seed is not None:
        # Seed NumPy, Torch, and Python RNGs for reproducibility.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    env = make_env(args.render)
    # obs is environment-specific; extract_board returns np.ndarray HxW
    obs, _ = env.reset(seed=args.seed)
    board = extract_board(obs)
    # extract_features returns shape (1, FEATURE_DIM); [0] -> (FEATURE_DIM,)
    prev_features = extract_features(board, lines_cleared=0)[0]

    # Q-networks map (batch, FEATURE_DIM) -> (batch, 1)
    policy = AfterStateQNetwork(FEATURE_DIM).to(device)
    target = AfterStateQNetwork(FEATURE_DIM).to(device)
    target.load_state_dict(policy.state_dict())
    # eval() switches layers like dropout/batchnorm to inference behavior.
    target.eval()
    # Adam optimizer maintains running moments per parameter.
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)
    # sequences: list of macro-actions, each is a list of int action codes
    sequences = build_action_sequences(args.max_shift)

    global_step = 0
    # global_step counts macro-actions across all episodes.
    # episode_rewards: list of per-episode returns (scalar floats)
    episode_rewards = []
    for episode in range(1, args.episodes + 1):
        seed = args.seed + episode if args.seed is not None else None
        obs, _ = env.reset(seed=seed)
        prev_features = extract_features(extract_board(obs), lines_cleared=0)[0]
        candidates = enumerate_after_states(env, sequences)

        terminated = False
        truncated = False
        steps = 0
        total_reward = 0.0

        while not (terminated or truncated):
            if not candidates:
                break

            # Batch feature extraction for all candidate after-states.
            # candidate_features: np.ndarray shape (num_candidates, FEATURE_DIM)
            candidate_features = np.stack([c["features"] for c in candidates], axis=0).astype(
                np.float32
            )
            # candidate_rewards: np.ndarray shape (num_candidates,)
            candidate_rewards = compute_candidate_rewards(prev_features, candidate_features)

            epsilon = epsilon_by_step(
                global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay
            )
            # action_idx: int index into candidates list
            action_idx = select_action(
                policy, candidate_features, candidate_rewards, device, epsilon, args.gamma
            )
            sequence = candidates[action_idx]["sequence"]

            # Execute the macro-action on the real env.
            # run_sequence returns (obs, reward_sum, done, info); reward_sum unused here
            obs, _, done, info = run_sequence(env, sequence)
            terminated = done
            truncated = False

            # Compute shaped reward from feature deltas.
            lines_cleared = extract_lines_cleared(info)
            # next_features: np.ndarray shape (FEATURE_DIM,)
            next_features = extract_features(extract_board(obs), lines_cleared)[0]
            # next_features[None, :] -> shape (1, FEATURE_DIM) for vectorized reward
            shaped_reward = compute_candidate_rewards(prev_features, next_features[None, :])[0]
            total_reward += shaped_reward

            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                truncated = True
                terminated = True

            if not (terminated or truncated):
                next_candidates = enumerate_after_states(env, sequences)
                next_candidate_features = (
                    # next_candidate_features: np.ndarray shape (num_next, FEATURE_DIM)
                    np.stack([c["features"] for c in next_candidates], axis=0).astype(np.float32)
                    if next_candidates
                    # Empty array with shape (0, FEATURE_DIM) when no candidates
                    else np.empty((0, FEATURE_DIM), dtype=np.float32)
                )
            else:
                next_candidates = []
                next_candidate_features = np.empty((0, FEATURE_DIM), dtype=np.float32)

            # Store transition: (after-state features, shaped reward, done flag, next candidates)
            buffer.push(next_features, shaped_reward, terminated or truncated, next_candidate_features)

            if len(buffer) >= args.start_training and len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                # loss: scalar torch.Tensor, mean squared TD error
                loss = compute_batch_loss(policy, target, batch, args.gamma, device)
                # Clear previous gradients before backprop.
                optimizer.zero_grad()
                # backprop computes gradients for all parameters involved in loss
                loss.backward()
                if args.max_grad_norm is not None:
                    # clip_grad_norm_ rescales gradients to limit their norm
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                # optimizer.step applies parameter updates
                optimizer.step()

            if global_step % args.target_update == 0:
                # Sync target network with policy network weights.
                target.load_state_dict(policy.state_dict())

            # Advance to the next after-state.
            prev_features = next_features
            candidates = next_candidates
            global_step += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        if args.log_interval and episode % args.log_interval == 0:
            recent = episode_rewards[-args.log_interval :]
            avg_return = float(np.mean(recent)) if recent else 0.0
            print(
                f"Episode {episode}/{args.episodes} avg_return={avg_return:.2f} "
                f"epsilon={epsilon:.3f}" # type: ignore
            )

        if args.save_interval and episode % args.save_interval == 0:
            save_checkpoint(args.save_path, policy, target, optimizer, episode)

    save_checkpoint(args.save_path, policy, target, optimizer, args.episodes)
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
