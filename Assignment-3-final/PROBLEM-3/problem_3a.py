import gymnasium as gym
import ale_py  
import cv2    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import random
import math
import matplotlib.pyplot as plt
import os

def run_random_episode(env_name):
    print(f"\n--- Running Random Episode with random agent for: {env_name} ---")
    
    # Create the environment
    
    env = gym.make(env_name,render_mode='human')
    
    # Reset the environment to get the first observation
    observation, info = env.reset(seed=42)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print("State Space", state_size)
    print("Action Space", action_size)
    
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    # Start the episode loop
    while not (terminated or truncated):
        # Take a random action
        action = env.action_space.sample()
        
        # Get the results from the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Print the reward at each step
        print(f"Step: {step_count},  Action: {action}, Reward: {reward}")

    # Episode is over
    print("--- Episode Finished ---")
    print(f"Total Steps Taken: {step_count}")
    print(f"Total Reward Gained: {total_reward}")
    
    env.close()


print("PROBLEM 1A : RANDOM AGENT")
run_random_episode("LunarLander-v3")
run_random_episode("CartPole-v1")

