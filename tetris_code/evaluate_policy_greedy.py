import cv2
import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
import policies

NB_EPISODES = 10

if __name__ == "__main__":
    #create the environement
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    #store the reward of each episodes
    reward_episodes = np.zeros(NB_EPISODES)
    #generate NB_EPISODES independent episodes using the given policy
    for episode in range(NB_EPISODES):
        #initialize the environement including the RNG
        env.reset()
        #keep track of episode end, total reward and time
        terminated = False
        total_reward = 0
        t = 0
        #loop until the game terminates i.e. the bricks reach to top of the screen
        while not terminated:
            t += 1
            #select action according to the given policy
            action = policies.policy_greedy(env) 
            #perform action, observe the reward and the next state
            observation, reward, terminated, truncated, info = env.step(action)
            #add the reward to the sum of rewards so far 
            total_reward = reward + total_reward # type: ignore
            #display the current score in the terminal
            print("Episode number" , 1+episode , "Score:",total_reward,end='\r', flush=True) 
        #when game terminates output the final score i.e. the sum of rewards across the episode 
        reward_episodes[episode] = total_reward
    #display the total reward of each episode
    print("Reward of episodes", reward_episodes)  
    #display the average reward with a confidence interval of the reward of episodes
    print("Average reward", np.mean(reward_episodes)," +/- ", np.sqrt((1/NB_EPISODES)*np.mean(reward_episodes ** 2)))
