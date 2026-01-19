import argparse
import os

import gymnasium as gym
from tetris_gymnasium.envs import Tetris
import numpy as np
import torch

from pg_cnn import PolicyNetwork, compute_returns, preprocess_observation, select_action


def make_env(render):
    if render:
        return gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    return gym.make("tetris_gymnasium/Tetris")


def save_checkpoint(path, policy, optimizer, episode, obs_shape, num_actions):
    payload = {
        "episode": episode,
        "obs_shape": tuple(obs_shape),
        "num_actions": int(num_actions),
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(payload, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a vanilla policy gradient CNN agent for Tetris.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--save-path", type=str, default="tetris_code/checkpoints/pg_cnn.pt")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--normalize-returns", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env = make_env(args.render)
    obs, _ = env.reset(seed=args.seed)
    obs_tensor = preprocess_observation(obs)
    policy = PolicyNetwork(obs_tensor.shape, env.action_space.n).to(device) # type: ignore
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    if args.seed is not None:
        env.action_space.seed(args.seed)

    episode_rewards = []
    for episode in range(1, args.episodes + 1):
        seed = args.seed + episode if args.seed is not None else None
        obs, _ = env.reset(seed=seed)
        log_probs = []
        rewards = []
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            obs_tensor = preprocess_observation(obs)
            action, log_prob = select_action(policy, obs_tensor, device, greedy=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                truncated = True

        if not rewards:
            print(f"Episode {episode}: empty episode, skipping update.")
            continue

        returns = compute_returns(rewards, args.gamma)
        if args.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_tensor = torch.stack(log_probs)
        loss = -(log_probs_tensor * returns.to(device)).sum()

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()

        episode_reward = float(np.sum(rewards))
        episode_rewards.append(episode_reward)

        if args.log_interval and episode % args.log_interval == 0:
            recent = episode_rewards[-args.log_interval :]
            avg_reward = float(np.mean(recent)) if recent else 0.0
            print(
                f"Episode {episode}/{args.episodes} "
                f"avg_reward={avg_reward:.2f} loss={loss.item():.4f}"
            )

        if args.save_interval and episode % args.save_interval == 0:
            save_checkpoint(
                args.save_path,
                policy,
                optimizer,
                episode,
                obs_tensor.shape,
                env.action_space.n, # type: ignore
            )

    save_checkpoint(
        args.save_path, policy, optimizer, args.episodes, obs_tensor.shape, env.action_space.n # type: ignore
    )
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
