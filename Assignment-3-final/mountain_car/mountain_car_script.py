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

    # Clip the gradients to a maximum value (e.g., 100)
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10.0)

    # Update the policy_net's weights
    optimizer.step()

    return loss.item()

# --- Hyperparameters ---
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-4
BUFFER_CAPACITY = 100000
MIN_BUFFER_SIZE_FOR_TRAINING = 20000
TARGET_UPDATE_FREQUENCY = 200

# Epsilon parameters
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = 30000  
steps_done = 0

# --- Initialization ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

env = gym.make("MountainCar-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create nets, optimizer, buffer, loss
policy_net = QNetwork(state_size, action_size).to(device)
target_net = QNetwork(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
criterion = nn.SmoothL1Loss()
buffer = ReplayBuffer(BUFFER_CAPACITY)

# --- Main Training Loop ---
num_episodes = 30000
episode_rewards = [] # To store scores for plotting
CHECKPOINT_PATH = 'mountain_dqn_checkpoint.pth'

start_episode = 0
steps_done = 0
losses = []


# --- Check for and Load Checkpoint ---
if os.path.exists(CHECKPOINT_PATH):
    print(f"--- Checkpoint found! Loading from {CHECKPOINT_PATH} ---")
    # Load the data from the file
    checkpoint = torch.load(CHECKPOINT_PATH,weights_only=False)
    
    # Restore the Policy Network's weights
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    
    # Restore the Target Network's weights
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    
    # Restore the Optimizer's state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    losses = checkpoint["losses"]
    episode_rewards = checkpoint["rewards"]
    
    # Restore the counters
    start_episode = checkpoint['episode'] + 1 # Start at the *next* episode
    steps_done = checkpoint['steps_done']
    
    # Restore the replay buffer
    if 'replay_buffer' in checkpoint:
        buffer = checkpoint['replay_buffer']
    
    print(f"--- Resuming training from episode {start_episode} ---")
    
else:
    print("--- No checkpoint found. Starting from scratch. ---")


for episode in range(start_episode,num_episodes):
    # Reset the environment
    state, info = env.reset(seed=42)

    total_reward = 0
    terminated = False
    truncated = False
    steps_this_episode = 0
    total_loss_this_episode = 0

    while not (terminated or truncated):
        # Select an action (epsilon-greedy)
        action = select_action(state, policy_net, env, device)
        
        # Take the action in the env
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # Store this experience in the replay buffer
        buffer.push(state, action, reward, next_state, terminated)

        # Set the new state for the next loop iteration
        state = next_state

        # Perform one step of optimization on the policy_net
        loss = optimize_model(buffer, policy_net, target_net, optimizer, criterion, device)

        # --- Add this block ---
        # Accumulate the loss
        if loss is not None:
            total_loss_this_episode += loss
            steps_this_episode += 1

        # Update the target network
        if steps_this_episode % TARGET_UPDATE_FREQUENCY == 0:
            print(f"*** Updating Target Network at episode {episode} ***")
            target_net.load_state_dict(policy_net.state_dict())
    
    
    # --- End of Episode ---
    # Calculate the average loss for the episode
    avg_loss = total_loss_this_episode / steps_this_episode if steps_this_episode > 0 else 100
    episode_rewards.append(total_reward)
    losses.append(avg_loss) 
    print()
    print(f"Episode {episode}, Avg. Loss: {avg_loss:.6f}, Total Reward: {total_reward}")

    
    # --- Checkpointing Logic ---
    # Save a checkpoint every 50 episodes
    if episode % 500 == 0:
        try:
            # CHECKPOINT_PATH = 'mountain_dqn_checkpoint.pth'
            print(f"--- Saving checkpoint at episode {episode} ---")
            
            # We create a dictionary to save all the important parts
            torch.save({
                'episode': episode,
                'steps_done': steps_done,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'replay_buffer': buffer,  # Optional: save the whole buffer,
                'losses': losses,
                'rewards': episode_rewards
            }, CHECKPOINT_PATH)
        except Exception as e:
            print("Error happended not able to save the checkpoint",e)
            print("")

print("--- Training Complete ---")
env.close()

# SAVE THE MODEL WEIGHTS
MODEL_PATH = 'mountaincar_dqn.pth'
torch.save(policy_net.state_dict(), MODEL_PATH)


# After the loop, plot the episode_rewards list
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.title('DQN Learning Curve for MountainCar-v0')
plt.xlabel('Episode')
plt.ylabel('Total Reward per Episode')
plt.grid(True)
plt.show()