import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tetris_gymnasium.envs import Tetris

from dqn_afterstate import (
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
LINE_CLEAR_REWARDS = {1: 1.0, 2: 3.0, 3: 6.0, 4: 12.0}


def make_env(render):
    """Create the Gymnasium Tetris environment."""
    if render:
        return gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    return gym.make("tetris_gymnasium/Tetris")


def save_checkpoint(path, policy, target, optimizer, episode):
    """Persist training state for later evaluation."""
    payload = {
        "episode": episode,
        "feature_dim": FEATURE_DIM,
        "model_state": policy.state_dict(),
        "target_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(payload, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN after-state agent with Dellacherie features."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--start-training", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=int, default=20000)
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--save-path", type=str, default="tetris_code/checkpoints/dqn_afterstate.pt"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    return parser.parse_args()


def epsilon_by_step(step, start, end, decay):
    """Exponential epsilon schedule."""
    if decay <= 0:
        return end
    mix = np.exp(-step / decay)
    return end + (start - end) * mix


def compute_candidate_rewards(prev_features, candidate_features):
    """
    Vectorized reward shaping:
      + line reward
      - holes created
      - bumpiness increase
      + survival bonus
    """
    lines = candidate_features[:, FEATURE_LINES]
    line_rewards = (
        (lines == 1) * LINE_CLEAR_REWARDS[1]
        + (lines == 2) * LINE_CLEAR_REWARDS[2]
        + (lines == 3) * LINE_CLEAR_REWARDS[3]
        + (lines == 4) * LINE_CLEAR_REWARDS[4]
    )

    # Penalize only *new* holes and *increased* bumpiness.
    holes_delta = np.maximum(0.0, candidate_features[:, FEATURE_HOLES] - prev_features[FEATURE_HOLES])
    bump_delta = np.maximum(
        0.0, candidate_features[:, FEATURE_BUMPINESS] - prev_features[FEATURE_BUMPINESS]
    )

    # Small survival bonus per placed piece.
    survival_bonus = 0.1
    shaped = line_rewards - 2.0 * holes_delta - 0.5 * bump_delta + survival_bonus
    return shaped.astype(np.float32)


def select_action(policy, candidate_features, candidate_rewards, device, epsilon, gamma):
    """Epsilon-greedy action selection using reward + gamma * V(after-state)."""
    if candidate_features.shape[0] == 0:
        raise RuntimeError("No candidate after-states available.")
    if random.random() < epsilon:
        return random.randrange(candidate_features.shape[0])

    # Batch all candidates for a single GPU/MPS forward pass.
    features_tensor = torch.from_numpy(candidate_features).to(device)
    rewards_tensor = torch.from_numpy(candidate_rewards).to(device)
    with torch.no_grad():
        values = policy(features_tensor).squeeze(-1)
    scores = rewards_tensor + gamma * values
    return int(torch.argmax(scores).item())


def compute_batch_loss(policy, target, batch, gamma, device):
    """DDQN loss: policy selects next state, target evaluates it."""
    states, rewards, dones, next_states = batch

    # Current after-state values from the policy network.
    state_tensor = torch.from_numpy(np.stack(states).astype(np.float32)).to(device)
    q_pred = policy(state_tensor).squeeze(-1)

    targets = []
    for reward, done, next_after_features in zip(rewards, dones, next_states):
        if done or getattr(next_after_features, "size", 0) == 0:
            targets.append(float(reward))
            continue

        next_tensor = torch.from_numpy(next_after_features.astype(np.float32)).to(device)
        with torch.no_grad():
            # DDQN: policy picks the best next after-state.
            next_q_policy = policy(next_tensor).squeeze(-1)
            best_idx = int(torch.argmax(next_q_policy).item())
            # Target network evaluates that choice.
            next_q_target = target(next_tensor).squeeze(-1)[best_idx].item()
        targets.append(float(reward) + gamma * next_q_target)

    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    return F.mse_loss(q_pred, target_tensor)


def main():
    args = parse_args()
    if args.device == "auto":
        # Prefer MPS for Apple Silicon, then CUDA, then CPU.
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    env = make_env(args.render)
    obs, _ = env.reset(seed=args.seed)
    board = extract_board(obs)
    prev_features = extract_features(board, lines_cleared=0)[0]

    policy = AfterStateQNetwork(FEATURE_DIM).to(device)
    target = AfterStateQNetwork(FEATURE_DIM).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)
    sequences = build_action_sequences(args.max_shift)

    global_step = 0
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
            candidate_features = np.stack([c["features"] for c in candidates], axis=0).astype(
                np.float32
            )
            candidate_rewards = compute_candidate_rewards(prev_features, candidate_features)

            epsilon = epsilon_by_step(
                global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay
            )
            action_idx = select_action(
                policy, candidate_features, candidate_rewards, device, epsilon, args.gamma
            )
            sequence = candidates[action_idx]["sequence"]

            # Execute the macro-action on the real env.
            obs, _, done, info = run_sequence(env, sequence)
            terminated = done
            truncated = False

            # Compute shaped reward from feature deltas.
            lines_cleared = extract_lines_cleared(info)
            next_features = extract_features(extract_board(obs), lines_cleared)[0]
            shaped_reward = compute_candidate_rewards(prev_features, next_features[None, :])[0]
            total_reward += shaped_reward

            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                truncated = True
                terminated = True

            if not (terminated or truncated):
                next_candidates = enumerate_after_states(env, sequences)
                next_candidate_features = (
                    np.stack([c["features"] for c in next_candidates], axis=0).astype(np.float32)
                    if next_candidates
                    else np.empty((0, FEATURE_DIM), dtype=np.float32)
                )
            else:
                next_candidates = []
                next_candidate_features = np.empty((0, FEATURE_DIM), dtype=np.float32)

            buffer.push(next_features, shaped_reward, terminated or truncated, next_candidate_features)

            if len(buffer) >= args.start_training and len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                loss = compute_batch_loss(policy, target, batch, args.gamma, device)
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

            if global_step % args.target_update == 0:
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
            avg_reward = float(np.mean(recent)) if recent else 0.0
            print(
                f"Episode {episode}/{args.episodes} avg_reward={avg_reward:.2f} "
                f"epsilon={epsilon:.3f}" # type: ignore
            )

        if args.save_interval and episode % args.save_interval == 0:
            save_checkpoint(args.save_path, policy, target, optimizer, episode)

    save_checkpoint(args.save_path, policy, target, optimizer, args.episodes)
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
