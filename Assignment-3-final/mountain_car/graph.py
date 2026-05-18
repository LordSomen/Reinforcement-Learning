import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to your checkpoint file
CHECKPOINT_PATH = '/Users/soumyajitpal/codes/IIT_HYD/SEM3/Reinforcement Learning/Assignment-3/mountain_car/mountain_dqn_checkpoint.pth'

def calculate_rolling_average(data, window_size):
    """Calculates a smooth, rolling average over the data."""
    # This uses convolution to create a sliding window average
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# --- Load the Checkpoint File ---
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
    exit()
    
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

# --- Extract the Data ---
if 'rewards' not in checkpoint or 'losses' not in checkpoint:
    print("Error: Checkpoint file is missing 'rewards' or 'losses' data.")
    exit()
    
episode_rewards = checkpoint['rewards']
losses = checkpoint['losses']

# Use a standard window size for the "n-episode mean"
n_window = 100

# --- Plot 1: Average Loss vs. Episode ---
# This plots the average loss you saved from each episode
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Average Loss per Episode', color='#d62728')
plt.title('Training Loss Curve (MountainCar-v0)')
plt.xlabel('Episode')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.savefig('mountaincar_training_loss_curve.png')
print("Saved 'mountaincar_training_loss_curve.png'")
plt.show()

# X-axis: "number of time steps (suitably scaled)"
# Y-axis: "mean n-episode reward" and "best mean reward"

print("Generating plot for Assignment 1b...")

# Get the x-axis: "number of time steps"
# For MountainCar-v0, reward is -1 per step.
# Therefore, episode length = -total_reward.
episode_lengths = [-r for r in episode_rewards]

# We create the cumulative sum of all steps taken
cumulative_timesteps = np.cumsum(episode_lengths)

# Get the y-axis: "mean n-episode reward"
# We calculate the rolling average
mean_n_rewards = calculate_rolling_average(episode_rewards, n_window)

# Get the y-axis: "best mean reward"
best_mean_reward = np.max(mean_n_rewards)

plt.figure(figsize=(12, 6))

# Plot the mean reward vs. the *time step* at the end of that window
# We must offset the x-axis (cumulative_timesteps) to align with the
# end of the rolling average window.
x_axis = cumulative_timesteps[n_window - 1:]
plt.plot(x_axis, mean_n_rewards, label=f'Mean {n_window}-Episode Reward', color='#1f77b4')

# Plot the best mean reward as a horizontal line
plt.axhline(best_mean_reward, color='red', linestyle='--', label=f'Best Mean Reward: {best_mean_reward:.2f}')

plt.title('Mean Reward vs. Total Time Steps (MountainCar-v0)')
plt.xlabel('Total Time Steps')
plt.ylabel('Mean Reward')
plt.legend()
plt.grid(True)
plt.savefig('mountaincar_timestep_reward_curve.png')
print("Saved 'mountaincar_timestep_reward_curve.png'")
plt.show()

print("All plots generated successfully.")