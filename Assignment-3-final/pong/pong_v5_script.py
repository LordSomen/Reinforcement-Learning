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



# --- Preprocessing Function ---
def preprocess_frame(frame):
    """
    Takes a raw Pong frame and
    preprocesses it to an 84x84 grayscale float.
    """
    # Convert to grayscale (if it's not already)
    # The raw 'ALE/Pong-v5' obs is [210, 160, 3]
    if len(frame.shape) > 2:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame
    
    # Crop out the score bar (top ~34 pixels) and bottom border
    cropped_frame = gray_frame[34:194, :] # Crop rows 34 to 194
    
    # Downsample (resize) to 84x84
    resized_frame = cv2.resize(cropped_frame, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values (to a 0.0 to 1.0 range)
    normalized_frame = np.array(resized_frame, dtype=np.float32) / 255.0
    
    # The final shape is [84, 84]
    return normalized_frame

# --- The CNN Q-Network ---
class PongQNetwork(nn.Module):
    def __init__(self, input_channels, action_size):
        super(PongQNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.flatten = nn.Flatten()
        
        # Calculate the flattened size after conv layers:
        # Input 84x84 -> conv1 (k=8,s=4) -> 20x20
        # 20x20 -> conv2 (k=4,s=2) -> 9x9
        # 9x9 -> conv3 (k=3,s=1) -> 7x7
        # Flattened size = 64 * 7 * 7 = 3136
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

# --- 3. Replay Buffer (Identical to MountainCar) ---
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

        # Convert to tensors on the fly
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self):
        return len(self.memory)

# --- 4. Select Action Function ---
# Epsilon parameters
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 200000  # Slower decay for Pong
steps_done = 0

def select_action(state, policy_net, env, device):
    global steps_done
    
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
              math.exp(-1. * steps_done / epsilon_decay)
    steps_done += 1
    
    if random.random() > epsilon:
        # EXPLOITATION
        with torch.no_grad():
            # 1. Convert NumPy state to the correct Tensor shape
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            # 2. Get Q-values
            q_values = policy_net(state_t)
            # 3. Choose best action
            # Note: Pong actions are 0, 2, 3. We map 0,1,2 -> 0,2,3
            action_index = q_values.argmax().item()
            if action_index == 1: return 2 # Map 1 to 2 (UP)
            if action_index == 2: return 3 # Map 2 to 3 (DOWN)
            return 0 # Map 0 to 0 (NOOP)
    else:
        # EXPLORATION
        # Return one of the three allowed actions
        return random.choice([0, 2, 3])

# --- 5. Optimize Model Function (Identical to MountainCar) ---
def optimize_model(buffer, policy_net, target_net, optimizer, criterion, device):
    # wait for minimum buffer to fill up
    if len(buffer) < MIN_BUFFER_SIZE_FOR_TRAINING:
        return None

    states_t, actions_t, rewards_t, next_states_t, dones_t = buffer.sample(BATCH_SIZE, device)
    
    # Map actions 0, 2, 3 back to 0, 1, 2 for indexing
    actions_t[actions_t == 2] = 1
    actions_t[actions_t == 3] = 2

    # --- Calculate X: The Q-values our policy_net PREDICTED ---
    all_q_values = policy_net(states_t)
    predicted_q_values = all_q_values.gather(1, actions_t.unsqueeze(1))

    # --- Calculate Y: The "Correct" Target Q-values ---
    with torch.no_grad():
        next_state_q_values = target_net(next_states_t)
        max_next_q = next_state_q_values.max(1)[0]
        target_q_values = rewards_t + (GAMMA * max_next_q * (1 - dones_t))
    
    # --- Calculate Loss and Optimize ---
    loss = criterion(predicted_q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # Clip the gradients to a maximum value 
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10.0)
    optimizer.step()
    # print("Loss :", loss)

    return loss.item()

# --- Hyperparameters & Initialization ---
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
BUFFER_CAPACITY = 100000 # Larger buffer for Pong
MIN_BUFFER_SIZE_FOR_TRAINING = 20000
TARGET_UPDATE_FREQUENCY = 1000 # Update target net every 1000 steps
ACTION_SIZE = 3 # We are only using 3 actions (0, 2, 3)
INPUT_CHANNELS = 4 # 4 stacked frames

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# We use the 'ALE/Pong-v5' environment
# We also use a wrapper to make the game end after one life
# 'frameskip=4' skips 3 frames for every 1 we process (speeds up game)
env = gym.make("ALE/Pong-v5", frameskip=4)
# env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=True, noop_max=30)

# Create nets, optimizer, buffer, loss
policy_net = PongQNetwork(INPUT_CHANNELS, ACTION_SIZE).to(device)
target_net = PongQNetwork(INPUT_CHANNELS, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.SmoothL1Loss()
buffer = ReplayBuffer(BUFFER_CAPACITY)

# --- 7. Main Training Loop ---
num_episodes = 2000 # Pong takes a *long* time
episode_rewards = []
frame_deque = collections.deque(maxlen=4)

CHECKPOINT_PATH = 'pong_dqn_checkpoint.pth'
start_episode = 0
steps_done = 0
losses = []


# --- Check for and Load Checkpoint ---
if os.path.exists(CHECKPOINT_PATH):
    print(f"--- Checkpoint found! Loading from {CHECKPOINT_PATH} ---")
    
    # 1. Load the data from the file
    checkpoint = torch.load(CHECKPOINT_PATH,weights_only=False)
    
    # 2. Restore the Policy Network's weights
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    
    # 3. Restore the Target Network's weights
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    
    # 4. Restore the Optimizer's state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    losses = checkpoint["losses"]
    episode_rewards = checkpoint["rewards"]
    
    # 5. Restore the counters
    start_episode = checkpoint['episode'] + 1 # Start at the *next* episode
    steps_done = checkpoint['steps_done']
    
    # 6. Optional: Restore the replay buffer
    if 'replay_buffer' in checkpoint:
        buffer = checkpoint['replay_buffer']
    
    print(f"--- Resuming training from episode {start_episode} ---")
    
else:
    print("--- No checkpoint found. Starting from scratch. ---")

for episode in range(start_episode,num_episodes):
    # 1. Reset env and get first frame
    raw_frame, info = env.reset(seed=42)
    processed_frame = preprocess_frame(raw_frame)
    
    # Fill the deque with 4 copies of the first frame
    for _ in range(4):
        frame_deque.append(processed_frame)
    
    # Our initial state is the stack of 4 identical frames
    state = np.array(frame_deque)
    
    total_reward = 0
    terminated = False
    truncated = False
    steps_this_episode = 0
    total_loss_this_episode = 0

    while not (terminated or truncated):
        # 2. Select an action (epsilon-greedy)
        action = select_action(state, policy_net, env, device)
        
        # 3. Take the action in the env
        new_raw_frame, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 4. Preprocess the new frame
        new_processed_frame = preprocess_frame(new_raw_frame)
        
        # 5. Append new frame to deque
        frame_deque.append(new_processed_frame)
        
        # 6. Create the 'next_state' from the updated deque
        next_state = np.array(frame_deque)

        # 7. Store this experience (using 'terminated', not 'truncated')
        buffer.push(state, action, reward, next_state, terminated)

        # 8. Set the new state for the next loop
        state = next_state

        # 9. Perform one step of optimization
        loss = optimize_model(buffer, policy_net, target_net, optimizer, criterion, device)

        # --- Add this block ---
        # Accumulate the loss
        if loss is not None:
            total_loss_this_episode += loss
            steps_this_episode += 1
            

        # 10. Update the target network
        if steps_done % TARGET_UPDATE_FREQUENCY == 0:
            print(f"*** Updating Target Network at step {steps_done} ***")
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
    if episode % 50 == 0:
        try:
            # CHECKPOINT_PATH = 'pong_dqn_checkpoint.pth'
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

# 1. SAVE THE MODEL WEIGHTS
MODEL_PATH = 'pong_dqn.pth'
torch.save(policy_net.state_dict(), MODEL_PATH)


# --- 8. Plot the Learning Curve ---
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.title('DQN Learning Curve for Pong-v5')
plt.xlabel('Episode')
plt.ylabel('Total Reward per Episode')
plt.grid(True)
plt.show()