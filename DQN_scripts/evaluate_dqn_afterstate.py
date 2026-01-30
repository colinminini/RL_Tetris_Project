import argparse

import cv2
import gymnasium as gym
import numpy as np
import torch
from tetris_gymnasium.envs import Tetris

from DQN_scripts.dqn_afterstate import (
    AfterStateQNetwork,
    FEATURE_BUMPINESS,
    FEATURE_DIM,
    FEATURE_HOLES,
    FEATURE_LINES,
    build_action_sequences,
    enumerate_after_states,
    extract_board,
    extract_features,
    extract_lines_cleared,
    run_sequence,
)

# Same shaping rewards as training for consistent behavior.
# Keys are line counts (int), values are shaped rewards (float).
LINE_CLEAR_REWARDS = {1: 1.0, 2: 3.0, 3: 6.0, 4: 12.0}


def make_env(render):
    """Create the Gymnasium Tetris environment."""
    # render=True enables human-mode rendering with scaled window.
    if render:
        return gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    return gym.make("tetris_gymnasium/Tetris")


def parse_args():
    # CLI for evaluation configuration.
    parser = argparse.ArgumentParser(description="Evaluate a DQN after-state Tetris agent.")
    # model-path: .pt file with model_state; episodes: number of evaluation runs.
    parser.add_argument(
        "--model-path", type=str, default="DQN_scripts/checkpoints/dqn_afterstate.pt"
    )
    parser.add_argument("--episodes", type=int, default=20)
    # seed: RNG seed; device: "auto" or explicit torch device string.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    # max-steps: optional cap on macro-actions per episode.
    parser.add_argument("--max-steps", type=int, default=None)
    # max-shift: number of left/right shifts to consider per macro-action.
    parser.add_argument("--max-shift", type=int, default=10)
    # gamma: discount factor for value estimates in action selection.
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def compute_candidate_rewards(prev_features, candidate_features):
    """Match the training reward shaping for evaluation."""
    # prev_features: np.ndarray shape (FEATURE_DIM,)
    # candidate_features: np.ndarray shape (num_candidates, FEATURE_DIM)
    lines = candidate_features[:, FEATURE_LINES]
    # lines: np.ndarray shape (num_candidates,)
    line_rewards = (
        (lines == 1) * LINE_CLEAR_REWARDS[1]
        + (lines == 2) * LINE_CLEAR_REWARDS[2]
        + (lines == 3) * LINE_CLEAR_REWARDS[3]
        + (lines == 4) * LINE_CLEAR_REWARDS[4]
    )
    # Penalize only newly created holes and increased bumpiness.
    holes_delta = np.maximum(0.0, candidate_features[:, FEATURE_HOLES] - prev_features[FEATURE_HOLES])
    bump_delta = np.maximum(
        0.0, candidate_features[:, FEATURE_BUMPINESS] - prev_features[FEATURE_BUMPINESS]
    )
    # shaped rewards: np.ndarray shape (num_candidates,)
    return (line_rewards - 2.0 * holes_delta - 0.5 * bump_delta + 0.1).astype(np.float32)


def select_action(policy, candidate_features, candidate_rewards, device, gamma):
    """Greedy selection: reward + gamma * V(after-state)."""
    # candidate_features: np.ndarray shape (num_candidates, FEATURE_DIM)
    # candidate_rewards: np.ndarray shape (num_candidates,)
    # torch.from_numpy creates a CPU tensor sharing memory with NumPy.
    features_tensor = torch.from_numpy(candidate_features).to(device)
    rewards_tensor = torch.from_numpy(candidate_rewards).to(device)
    # torch.no_grad disables gradient tracking for inference-time forward pass.
    with torch.no_grad():
        # policy output: (num_candidates, 1) -> squeeze -> (num_candidates,)
        values = policy(features_tensor).squeeze(-1)
    # scores: torch.Tensor shape (num_candidates,)
    scores = rewards_tensor + gamma * values
    # argmax returns index of the best candidate
    return int(torch.argmax(scores).item())


def render_step(env):
    """Render one frame and let the GUI event loop update."""
    env.render()
    # cv2.waitKey lets the window process events; 1 ms delay.
    cv2.waitKey(1)


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

    env = make_env(args.render)
    obs, _ = env.reset(seed=args.seed)
    # extract_features returns shape (1, FEATURE_DIM); [0] -> (FEATURE_DIM,)
    prev_features = extract_features(extract_board(obs), lines_cleared=0)[0]

    policy = AfterStateQNetwork(FEATURE_DIM).to(device)
    # torch.load returns a dict of tensors; map_location moves to chosen device.
    checkpoint = torch.load(args.model_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state"])
    # eval() switches layers like dropout/batchnorm to inference behavior.
    policy.eval()

    sequences = build_action_sequences(args.max_shift)
    # reward_episodes: np.ndarray shape (num_episodes,), per-episode env returns
    reward_episodes = np.zeros(args.episodes, dtype=np.float32)

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

            # candidate_features: np.ndarray shape (num_candidates, FEATURE_DIM)
            candidate_features = np.stack([c["features"] for c in candidates], axis=0).astype(
                np.float32
            )
            # candidate_rewards: np.ndarray shape (num_candidates,)
            candidate_rewards = compute_candidate_rewards(prev_features, candidate_features)
            action_idx = select_action(
                policy, candidate_features, candidate_rewards, device, args.gamma
            )

            sequence = candidates[action_idx]["sequence"]
            # run_sequence returns (obs, reward_sum, done, info)
            obs, reward_sum, done, info = run_sequence(
                env, sequence, render_fn=render_step if args.render else None
            )
            terminated = done

            lines_cleared = extract_lines_cleared(info)
            # next_features: np.ndarray shape (FEATURE_DIM,)
            next_features = extract_features(extract_board(obs), lines_cleared)[0]
            # total_reward: scalar float, running sum of raw env rewards (env return)
            total_reward += reward_sum
            # Match the greedy policy evaluation score display (env return).
            print(
                "Episode number",
                episode,
                "Env return:",
                total_reward,
                end="\r",
                flush=True,
            )

            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                truncated = True

            if not (terminated or truncated):
                candidates = enumerate_after_states(env, sequences)
                # Use next_features as prev_features for the next decision.
                prev_features = next_features

        reward_episodes[episode - 1] = total_reward

    print("Env return of episodes", reward_episodes)
    print(
        "Average env return",
        np.mean(reward_episodes),
        " +/- ",
        # Root-mean-square of episode returns (not standard deviation).
        np.sqrt((1 / args.episodes) * np.mean(reward_episodes**2)),
    )


if __name__ == "__main__":
    main()
