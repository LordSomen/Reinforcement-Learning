import gymnasium as gym
import ale_py
import torch
import sys
import random
import collections
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import os  # <-- 1. Import os to check for files

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # We store the experience as a tuple
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size, device):
        """
        Randomly samples a batch and converts it to Tensors on the correct device.
        """
        # Sample a list of experience-tuples
        experiences = random.sample(self.memory, batch_size)

        # "Unzip" the batch of tuples into separate tuples
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert each tuple of data into a PyTorch Tensor
        
        # We use np.vstack to stack states/next_states before tensor conversion
        states_t = torch.tensor(np.vstack(states), dtype=torch.float32).to(device)
        next_states_t = torch.tensor(np.vstack(next_states), dtype=torch.float32).to(device)
        
        # Actions are indices, so they should be long integers
        actions_t = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

        # Return the 5 tensors
        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self):
        return len(self.memory)



# (The QNetwork class definition remains the same)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        q_values = self.output_layer(x)
        return q_values

# --- 2. MODIFIED watch_trained_agent function ---
def watch_trained_agent(model_path, checkpoint_path, env_name="MountainCar-v0"):
    """
    Loads a trained model and runs one episode with rendering.
    Tries to load from model_path first, then falls back to checkpoint_path.
    """
    #  Setup the environment and network
    env = gym.make(env_name, render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    policy_net = QNetwork(state_size, action_size).to(device)

    if os.path.exists(model_path):
        print(f"--- Loading final model from {model_path} ---")
        # The final model file IS the state_dict
        policy_net.load_state_dict(torch.load(model_path))
        
    elif os.path.exists(checkpoint_path):
        print(f"--- Final model not found. Loading checkpoint from {checkpoint_path} ---")
        # The checkpoint is a DICTIONARY, we need to get the state_dict FROM it
        checkpoint = torch.load(checkpoint_path,weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        
    else:
        print(f"--- ERROR: No saved model or checkpoint found to watch. ---")
        print(f"Tried: {model_path}")
        print(f"And: {checkpoint_path}")
        env.close()
        return  

    policy_net.eval() 

    print(f"--- Watching trained agent in {env_name} ---")
    
    # Run one full episode
    state, info = env.reset()
    terminated, truncated = False , False
    total_reward = 0

    while not (terminated or truncated):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state_t)
            action = q_values.argmax().item()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

    print(f"--- Episode finished. Total Reward: {total_reward} ---")
    env.close()

# --- Call the function AFTER your plotting code ---
MODEL_PATH = '/Users/soumyajitpal/codes/IIT_HYD/SEM3/Reinforcement Learning/Assignment-3/mountain_car/mountaincar_dqn.pth'
CHECKPOINT_PATH = '/Users/soumyajitpal/codes/IIT_HYD/SEM3/Reinforcement Learning/Assignment-3/mountain_car/mountain_dqn_checkpoint.pth' 

# Pass BOTH paths to the function
watch_trained_agent(MODEL_PATH, CHECKPOINT_PATH)
