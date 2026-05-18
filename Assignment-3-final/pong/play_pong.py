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
import os  
import cv2

def preprocess_frame(frame):
    """
    Takes a raw Pong frame (e.g., 210x160x3) and
    preprocesses it to an 84x84 grayscale float.
    """
    # Convert to grayscale (if it's not already)
    # The raw 'ALE/Pong-v5' obs is [210, 160, 3]
    if len(frame.shape) > 2:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame
    
    cropped_frame = gray_frame[34:194, :] # Crop rows 34 to 194

    resized_frame = cv2.resize(cropped_frame, (84, 84), interpolation=cv2.INTER_AREA)

    normalized_frame = np.array(resized_frame, dtype=np.float32) / 255.0

    return normalized_frame

class PongQNetwork(nn.Module):
    def __init__(self, input_channels, action_size):
        super(PongQNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.flatten = nn.Flatten()
 
        self.fc1 = nn.Linear(3136, 512)
        self.output_layer = nn.Linear(512, action_size)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        q_values = self.output_layer(x)
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # We store the *NumPy arrays* in the buffer to save memory
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size, device):
        experiences = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self):
        return len(self.memory)

def watch_trained_agent(model_path, checkpoint_path, env_name="ALE/Pong-v5"):
    
    # --- Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # We MUST use render_fps to see the game at a normal speed
    env = gym.make(env_name, render_mode="human")
    
    # --- Correctly initialize the network ---
    INPUT_CHANNELS = 4
    ACTION_SIZE = 3 # (0, 2, 3)
    
    policy_net = PongQNetwork(INPUT_CHANNELS, ACTION_SIZE).to(device)

    # --- Loading Logic ---
    if os.path.exists(model_path):
        print(f"--- Loading final model from {model_path} ---")
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        
    elif os.path.exists(checkpoint_path):
        print(f"--- Final model not found. Loading checkpoint from {checkpoint_path} ---")
        # Add weights_only=False in case you saved the buffer
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device) 
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        
    else:
        print(f"--- ERROR: No saved model or checkpoint found. ---")
        env.close()
        return

    policy_net.eval()
    frame_deque = collections.deque(maxlen=4)

    print(f"--- 🚀 Watching trained agent in {env_name} ---")
    
    # --- Run one full episode ---
    raw_frame, info = env.reset()
    processed_frame = preprocess_frame(raw_frame)
    
    # Fill the deque with 4 copies of the first frame
    for _ in range(4):
        frame_deque.append(processed_frame)
    
    state = np.array(frame_deque)
    terminated, truncated = False, False
    total_reward = 0

    while not (terminated or truncated):
        with torch.no_grad():
            # Convert state to tensor
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get Q-values
            q_values = policy_net(state_t)
            
            # --- Correct Action Selection ---
            # Get network's action index (0, 1, or 2)
            action_index = q_values.argmax().item()
            
            # Map to Pong's real actions (0, 2, or 3)
            if action_index == 1: action = 2 # Map 1 to 2 (UP)
            elif action_index == 2: action = 3 # Map 2 to 3 (DOWN)
            else: action = 0 # Map 0 to 0 (NOOP)
        
        # --- Correct Frame Processing Loop ---
        new_raw_frame, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        new_processed_frame = preprocess_frame(new_raw_frame)
        frame_deque.append(new_processed_frame)
        
        # The new state is the updated stack
        state = np.array(frame_deque)

    print(f"--- 🏁 Episode finished. Total Reward: {total_reward} ---")
    env.close()

# --- Call the function---
MODEL_PATH = '/Users/soumyajitpal/codes/IIT_HYD/SEM3/Reinforcement Learning/Assignment-3/pong/pong_dqn.pth'
CHECKPOINT_PATH = '/Users/soumyajitpal/codes/IIT_HYD/SEM3/Reinforcement Learning/Assignment-3/pong/pong_dqn_checkpoint.pth'  

# Pass BOTH paths to the function
watch_trained_agent(MODEL_PATH, CHECKPOINT_PATH)