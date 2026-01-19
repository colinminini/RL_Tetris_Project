import argparse

import gymnasium as gym
from tetris_gymnasium.envs import Tetris
import numpy as np
import torch

from pg_cnn import PolicyNetwork, preprocess_observation, select_action


def make_env(render):
    if render:
        return gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    return gym.make("tetris_gymnasium/Tetris")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy gradient CNN agent.")
    parser.add_argument("--model-path", type=str, default="tetris_code/checkpoints/pg_cnn.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    env = make_env(args.render)
    obs, _ = env.reset(seed=args.seed)
    obs_tensor = preprocess_observation(obs)
    policy = PolicyNetwork(obs_tensor.shape, env.action_space.n).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()

    rewards = []
    with torch.no_grad():
        for episode in range(1, args.episodes + 1):
            seed = args.seed + episode if args.seed is not None else None
            obs, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            steps = 0
            total_reward = 0.0
            while not (terminated or truncated):
                if args.render:
                    env.render()
                obs_tensor = preprocess_observation(obs)
                action, _ = select_action(
                    policy, obs_tensor, device, greedy=not args.stochastic
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                if args.max_steps is not None and steps >= args.max_steps:
                    truncated = True
            rewards.append(total_reward)
            print(f"Episode {episode} reward={total_reward:.2f}")

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    print(
        f"Average reward={rewards_arr.mean():.2f} +/- {rewards_arr.std():.2f} "
        f"(n={len(rewards_arr)})"
    )


if __name__ == "__main__":
    main()
