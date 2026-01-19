import sys
import cv2
import gymnasium as gym
from tetris_gymnasium.envs import Tetris

if __name__ == "__main__":
    # Create an instance of Tetris
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    env.reset(seed=42)

    # Main game loop
    terminated = False
    total_reward = 0
    while not terminated:
        # Render the current state of the game as text
        env.render()
        # Pick an action from user input mapped to the keyboard
        action = None
        while action is None:
            key = cv2.waitKey(1)

            if key == ord("q"):
                action = env.unwrapped.actions.move_left # pyright: ignore[reportAttributeAccessIssue]
            elif key == ord("d"):
                action = env.unwrapped.actions.move_right # type: ignore
            elif key == ord("s"):
                action = env.unwrapped.actions.move_down # type: ignore
            elif key == ord("j"):
                action = env.unwrapped.actions.rotate_counterclockwise # type: ignore
            elif key == ord("k"):
                action = env.unwrapped.actions.rotate_clockwise # type: ignore
            elif key == ord(" "):
                action = env.unwrapped.actions.hard_drop    # type: ignore
            elif key == ord("q"):
                action = env.unwrapped.actions.swap # type: ignore
            elif key == ord("r"):
                env.reset(seed=42)
                break

            if (
                cv2.getWindowProperty(env.unwrapped.window_name, cv2.WND_PROP_VISIBLE) # type: ignore
                == 0
            ):
                sys.exit()

        # Perform the action
        observation, reward, terminated, truncated, info = env.step(action)
        #Keep track of the total reward
        total_reward = reward + total_reward # type: ignore
        #Display the current score in the terminal
        print("Score:",total_reward,end='\r', flush=True) 
    # Game over
    print("Game Over!")
