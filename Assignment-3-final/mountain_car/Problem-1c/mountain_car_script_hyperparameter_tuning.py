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


# --- Hyperparameters  ---
BATCH_SIZE = 128
GAMMA = 0.99
BUFFER_CAPACITY = 10000
MIN_BUFFER_SIZE_FOR_TRAINING = 2000 
TARGET_UPDATE_FREQUENCY = 10 

# --- Epsilon parameters  ---
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = 30000  
steps_done = 0

def select_action(state, policy_net, env, device):
    global steps_done
    
    # Calculate the current epsilon value
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
              math.exp(-1. * steps_done / epsilon_decay)
    steps_done += 1
    
    # Get a random number
    sample = random.random()

    if sample > epsilon:
        # EXPLOITATION: Use the network to pick the best action
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            best_action = policy_net(state_t).argmax().item()
        return best_action
    else:
        # EXPLORATION: Pick a random action
        return env.action_space.sample()


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size, device):
        """
        Randomly samples a batch and converts it to Tensors on the correct device.
        """
        # Sample a list of experience-tuples
        experiences = random.sample(self.memory, batch_size)

        # 2. "Unzip" the batch of tuples into separate tuples
        states, actions, rewards, next_states, dones = zip(*experiences)

        # We use np.vstack to stack states/next_states before tensor conversion
        # This creates a tensor of shape [batch_size, state_dim]
        states_t = torch.tensor(np.vstack(states), dtype=torch.float32).to(device)
        next_states_t = torch.tensor(np.vstack(next_states), dtype=torch.float32).to(device)
        
        # Actions are indices, so they should be long integers
        actions_t = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialize the network.
        
        Args:
            state_size (int): The number of inputs (2 for MountainCar)
            action_size (int): The number of outputs (3 for MountainCar)
        """
        super(QNetwork, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(state_size, 64)  
        self.layer2 = nn.Linear(64, 64)          
        self.output_layer = nn.Linear(64, action_size) 

    def forward(self, state):
        """
        Defines the forward pass of the network.
        
        Args:
            state (torch.Tensor): The input state tensor
        
        Returns:
            torch.Tensor: The Q-values for each action
        """
        # Pass state through layer 1, then apply ReLU activation
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        
        # Pass through the output layer (no activation here)
        q_values = self.output_layer(x)
        
        return q_values

def optimize_model(buffer, policy_net, target_net, optimizer, criterion, device):
    """
    Performs one step of learning:
    1. Samples a batch from the buffer
    2. Calculates the loss
    3. Updates the policy_net
    """
    # Don't try to learn if the buffer is not full enough
    if len(buffer) < MIN_BUFFER_SIZE_FOR_TRAINING:
        return None

    # Sample a batch from the replay buffer
    states_t, actions_t, rewards_t, next_states_t, dones_t = buffer.sample(BATCH_SIZE, device)

    # --- Calculate X: The Q-values our policy_net PREDICTED ---
    
    # Get Q-values for ALL actions from the policy_net
    all_q_values = policy_net(states_t)
    
    # Use .gather() to select only the Q-value for the action we ACTUALLY took
    predicted_q_values = all_q_values.gather(1, actions_t.unsqueeze(1))

    # --- Calculate Y: The "Correct" Target Q-values ---
    # We use .no_grad() because we don't need gradients for the target_net
    with torch.no_grad():
        # Get the target_net's Q-values for the *next* state
        next_state_q_values = target_net(next_states_t)
        
        # Find the BEST Q-value from the next state (this is the "max" part)
        max_next_q = next_state_q_values.max(1)[0]

        # Calculate the Bellman equation: Y = Reward + Gamma * Max_Next_Q
        target_q_values = rewards_t + (GAMMA * max_next_q * (1 - dones_t))

    # --- Calculate Loss and Optimize ---

    # Calculate the Loss (the error between our prediction X and target Y)
    loss = criterion(predicted_q_values, target_q_values.unsqueeze(1))

    # Clear old gradients
    optimizer.zero_grad()

    # Calculate new gradients (backpropagation)
    loss.backward()

    # Clip the gradients to a maximum value 
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10.0)
    
    # Update the policy_net's weights
    optimizer.step()

    return loss.item()

def run_experiment(learning_rate_to_test):
    print(f"\n--- STARTING EXPERIMENT: LR = {learning_rate_to_test} ---")
    # --- Initialization ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    env = gym.make("MountainCar-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # --- Re-initialize everything for this run ---
    # (Assuming your QNetwork is the new one with BatchNorm)
    policy_net = QNetwork(state_size, action_size).to(device)
    target_net = QNetwork(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate_to_test, amsgrad=True)
    criterion = nn.SmoothL1Loss()
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    # --- Training Loop ---
    num_episodes = 5000
    
    # --- Lists to store results ---
    episode_rewards = [] 
    episode_avg_losses = [] 

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        terminated, truncated = False, False
        
        # --- Loss tracking variables ---
        total_loss_this_episode = 0.0
        steps_this_episode = 0

        while not (terminated or truncated):
            action = select_action(state, policy_net, env, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            
            policy_net.train()
            loss = optimize_model(buffer, policy_net, target_net, optimizer, criterion, device)
            
            #  Accumulate loss
            if loss is not None:
                total_loss_this_episode += loss
                steps_this_episode += 1
            
        # End of Episode 
        episode_rewards.append(total_reward)
        
        # Calculate and store avg loss 
        avg_loss = total_loss_this_episode / steps_this_episode if steps_this_episode > 0 else 0
        episode_avg_losses.append(avg_loss)

        if episode % 50 == 0:
             # --- Updated print ---
             print(f"  [LR={learning_rate_to_test}] Episode {episode}, Reward: {total_reward}, AvgLoss: {avg_loss:.4f}")

        if episode % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print(f"--- FINISHED EXPERIMENT: LR = {learning_rate_to_test} ---")
    env.close()
    
    # --- Return both lists ---
    return episode_rewards, episode_avg_losses

if __name__ == "__main__":  
    # --- List of learning rates to test ---
    learning_rates_to_test = [1e-5, 1e-4, 5e-4, 1e-3]  
    # --- Dictionaries to store all our results ---
    all_rewards = {}
    all_losses = {}
    for lr in learning_rates_to_test:
        # --- Capture both return values ---
        rewards_list, losses_list = run_experiment(learning_rate_to_test=lr)
        # --- Store in separate dicts ---
        all_rewards[lr] = rewards_list
        all_losses[lr] = losses_list
    print("\n--- All experiments complete. Plotting... ---")
    # --- Ploting Rewrds ---
    colors = ['#1f77b4',  # blue
              '#f0e442',  # light yellow
              '#9467bd',  # violet
              '#d62728']  # red
    
    all_rewards = {}
    all_losses = {}

    for lr in learning_rates_to_test:
        rewards_list, losses_list = run_experiment(learning_rate_to_test=lr)
        all_rewards[lr] = rewards_list
        all_losses[lr] = losses_list

    print("\n--- All experiments complete. Plotting... ---")

    plt.figure(figsize=(12, 8))
    
    # --- Loop over both items and colors ---
    for (lr, rewards), color in zip(all_rewards.items(), colors):
        # Use the 'color=' argument
        plt.plot(rewards, label=f'Learning Rate = {lr}', color=color)

    plt.title('DQN Learning Rate Comparison on MountainCar-v0 (Reward)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('problem_1c_reward_comparison.png')
    plt.show()

    # --- Plotting Losses ---
    plt.figure(figsize=(12, 8))
    
    # --- loop over both items and colors ---
    for (lr, losses), color in zip(all_losses.items(), colors):
        # Use the 'color=' argument
        plt.plot(losses[10:], label=f'Learning Rate = {lr}', color=color) 
        
    plt.title('DQN Learning Rate Comparison on MountainCar-v0 (Average Loss)')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('problem_1c_loss_comparison.png')
    plt.show()