import cv2
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
import policies

if __name__ == "__main__":
    #create the environement
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    #initialize the environement including the RNG
    env.reset()
    #keep track of episode end, total reward and time
    terminated = False
    total_reward = 0
    t = 0
    #loop until the game terminates i.e. the bricks reach to top of the screen
    while not terminated:
        t += 1
        #render the GUI
        env.render()
        #select action according to the given policy
        action = policies.policy_greedy(env) 
        #perform action, observe the reward and the next state
        observation, reward, terminated, truncated, info = env.step(action)
        #wait to be able to see the movement
        key = cv2.waitKey(1) 
        #add the reward to the sum of rewards so far 
        total_reward = reward + total_reward # type: ignore
        #display the current score in the terminal
        print("Score:",total_reward,end='\r', flush=True) 
    #when game terminates output the final score i.e. the sum of rewards across the episode 
    print("Game Over!")
    print("Final Score:",total_reward)
