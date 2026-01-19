import argparse

import cv2
import gymnasium as gym
import numpy as np
import torch
from tetris_gymnasium.envs import Tetris

from dqn_afterstate import (
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
LINE_CLEAR_REWARDS = {1: 1.0, 2: 3.0, 3: 6.0, 4: 12.0}


def make_env(render):
    """Create the Gymnasium Tetris environment."""
    if render:
        return gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    return gym.make("tetris_gymnasium/Tetris")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DQN after-state Tetris agent.")
    parser.add_argument(
        "--model-path", type=str, default="tetris_code/checkpoints/dqn_afterstate.pt"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def compute_candidate_rewards(prev_features, candidate_features):
    """Match the training reward shaping for evaluation."""
    lines = candidate_features[:, FEATURE_LINES]
    line_rewards = (
        (lines == 1) * LINE_CLEAR_REWARDS[1]
        + (lines == 2) * LINE_CLEAR_REWARDS[2]
        + (lines == 3) * LINE_CLEAR_REWARDS[3]
        + (lines == 4) * LINE_CLEAR_REWARDS[4]
    )
    holes_delta = np.maximum(0.0, candidate_features[:, FEATURE_HOLES] - prev_features[FEATURE_HOLES])
    bump_delta = np.maximum(
        0.0, candidate_features[:, FEATURE_BUMPINESS] - prev_features[FEATURE_BUMPINESS]
    )
    return (line_rewards - 2.0 * holes_delta - 0.5 * bump_delta + 0.1).astype(np.float32)


def select_action(policy, candidate_features, candidate_rewards, device, gamma):
    """Greedy selection: reward + gamma * V(after-state)."""
    features_tensor = torch.from_numpy(candidate_features).to(device)
    rewards_tensor = torch.from_numpy(candidate_rewards).to(device)
    with torch.no_grad():
        values = policy(features_tensor).squeeze(-1)
    scores = rewards_tensor + gamma * values
    return int(torch.argmax(scores).item())


def render_step(env):
    """Render one frame and let the GUI event loop update."""
    env.render()
    cv2.waitKey(1)


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

    env = make_env(args.render)
    obs, _ = env.reset(seed=args.seed)
    prev_features = extract_features(extract_board(obs), lines_cleared=0)[0]

    policy = AfterStateQNetwork(FEATURE_DIM).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()

    sequences = build_action_sequences(args.max_shift)
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

            candidate_features = np.stack([c["features"] for c in candidates], axis=0).astype(
                np.float32
            )
            candidate_rewards = compute_candidate_rewards(prev_features, candidate_features)
            action_idx = select_action(
                policy, candidate_features, candidate_rewards, device, args.gamma
            )

            sequence = candidates[action_idx]["sequence"]
            obs, reward_sum, done, info = run_sequence(
                env, sequence, render_fn=render_step if args.render else None
            )
            terminated = done

            lines_cleared = extract_lines_cleared(info)
            next_features = extract_features(extract_board(obs), lines_cleared)[0]
            total_reward += reward_sum
            # Match the greedy policy evaluation score display.
            print(
                "Episode number",
                episode,
                "Score:",
                total_reward,
                end="\r",
                flush=True,
            )

            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                truncated = True

            if not (terminated or truncated):
                candidates = enumerate_after_states(env, sequences)
                prev_features = next_features

        reward_episodes[episode - 1] = total_reward

    print("Reward of episodes", reward_episodes)
    print(
        "Average reward",
        np.mean(reward_episodes),
        " +/- ",
        np.sqrt((1 / args.episodes) * np.mean(reward_episodes**2)),
    )


if __name__ == "__main__":
    main()
