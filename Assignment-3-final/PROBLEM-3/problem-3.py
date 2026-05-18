import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import argparse  # For command-line options
import collections

# --- The Actor: Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        logits = self.output_layer(x)
        # Return a distribution we can sample from
        return Categorical(logits=logits)

# --- The Critic: Value/Baseline Network ---
# This network learns the "baseline" V(s)
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        # Output is a single number: the estimated state value
        self.output_layer = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        value = self.output_layer(x)
        return value

# --- Helper Function to Calculate Rewards-to-Go ---
def calculate_rewards(rewards, gamma, use_reward_to_go):
    """
    Calculates the discounted rewards for a trajectory.
    This implements both Formulation 1 and 2 from the assignment.
    """
    T = len(rewards)
    discounted_rewards = np.zeros(T)
    
    if use_reward_to_go:
        # Formulation 2: Reward-to-Go 
        running_add = 0
        for t in reversed(range(T)):
            running_add = rewards[t] + gamma * running_add
            discounted_rewards[t] = running_add
    else:
        # Formulation 1: Total reward of the trajectory 
        # Calculate the total discounted reward G_0
        total_reward = 0
        for t in reversed(range(T)):
            total_reward = rewards[t] + gamma * total_reward
        # Assign the same G_0 to all steps
        discounted_rewards[:] = total_reward
        
    return discounted_rewards

# --- Main Training Function ---
def main(args):
    
    # --- Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Init Actor (Policy) and Critic (Value) networks
    policy_net = PolicyNetwork(state_size, action_size).to(device)
    value_net = ValueNetwork(state_size).to(device)
    
    # Init two separate optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.lr)
    
    # List to store rewards for plotting
    all_episode_rewards = []
    
    print(f"--- Starting Training on {args.env_name} ---")
    print(f"Reward-to-Go: {args.use_reward_to_go}, Advantage Norm: {args.use_advantage_norm}")

    # --- Main Training Loop ---
    # PG trains in "iterations" or "batches"
    for iteration in range(args.num_iterations):
        
        # --- Batch Collection ---
        # We collect one full trajectory (episode) per iteration
        # In a more advanced implementation, this would be a "batch"
        # of many trajectories 
        batch_rewards = []
        batch_log_probs = []
        batch_states = []
        
        state, info = env.reset()
        terminated = False
        truncated = False
        # You could also write: terminated, truncated = False, False
        episode_reward = 0

        # --- Episode Loop ---
        while not (terminated or truncated):
            state_t = torch.tensor(state, dtype=torch.float32).to(device)
            batch_states.append(state_t)
            
            # Run the Policy Network (Actor)
            action_dist = policy_net(state_t)
            
            # Sample an action
            action = action_dist.sample()
            
            # Get the log-probability of that action
            log_prob = action_dist.log_prob(action)
            batch_log_probs.append(log_prob)
            
            # Take the action in the environment
            action_item = action.item()
            next_state, reward, terminated, truncated, info = env.step(action_item)
            
            batch_rewards.append(reward)
            episode_reward += reward
            
            state = next_state
            
        all_episode_rewards.append(episode_reward)
        
        # --- End of Episode: Policy Update ---
        
        # Convert lists to tensors
        log_probs_t = torch.stack(batch_log_probs).to(device)
        states_t = torch.stack(batch_states).to(device)

        # Calculate Rewards (Psi_t)
        # This is G_t (Reward-to-Go) or G_0 (Total Reward)
        rewards_to_go_t = torch.tensor(
            calculate_rewards(batch_rewards, args.gamma, args.use_reward_to_go),
            dtype=torch.float32
        ).to(device)
        
        # Calculate Advantages (A_t)
        # A_t = G_t - V(s_t)
        state_values_v = value_net(states_t).squeeze()
        advantages_t = rewards_to_go_t - state_values_v
        
        # Advantage Normalization 
        if args.use_advantage_norm:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
        # Calculate Actor (Policy) Loss
        # We must .detach() advantages so that the Actor's loss
        # doesn't try to backpropagate into the Critic
        actor_loss = -(log_probs_t * advantages_t.detach()).mean()
        
        # Calculate Critic (Value) Loss
        # The critic is trained to predict the Reward-to-Go (G_t)
        critic_loss = F.mse_loss(state_values_v, rewards_to_go_t)
        
        # Optimize Actor
        policy_optimizer.zero_grad()
        actor_loss.backward()
        policy_optimizer.step()
        
        # Optimize Critic
        value_optimizer.zero_grad()
        critic_loss.backward()
        value_optimizer.step()

        # --- Logging ---
        if iteration % 20 == 0:
            print(f"Iteration {iteration}, Last Reward: {episode_reward:.2f}, Avg. Reward (Last 100): {np.mean(all_episode_rewards[-100:]):.2f}")
            
    env.close()
    
    # Define the save path
    MODEL_SAVE_PATH = f'pg_policy_net_{args.env_name}.pth'
    
    # Save the Policy Network's weights
    print(f"--- Training Complete. Saving model to {MODEL_SAVE_PATH} ---")
    torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)

    # --- 5. Plot Learning Curve ---
    # We smooth the curve to make it easier to read
    window_size = 50
    smoothed_rewards = [np.mean(all_episode_rewards[i-window_size:i]) for i in range(window_size, len(all_episode_rewards))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards)
    plt.title(f'PG on {args.env_name} (R2G: {args.use_reward_to_go}, AdvNorm: {args.use_advantage_norm})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward (Window 50)')
    plt.grid(True)
    plt.savefig(f'pg_curve_{args.env_name}_r2g{args.use_reward_to_go}_norm{args.use_advantage_norm}.png')
    print(f"--- Training Complete. Plot saved to pg_curve_{args.env_name}_...png ---")

# --- Argparse for Command-Line Options  ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Policy Gradient with Variance Reduction')
    
    # Environment and training args
    parser.add_argument('--env-name', type=str, default='CartPole-v1', help='Gym environment name (e.g., CartPole-v1, LunarLander-v3)')
    parser.add_argument('--num-iterations', type=int, default=1500, help='Number of iterations (episodes) to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizers')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    # Assignment-specific boolean flags 
    # For --use-reward-to-go
    parser.add_argument('--use-reward-to-go', action='store_true', default=True, help='Use reward-to-go (Formulation 2)')
    parser.add_argument('--no-reward-to-go', action='store_false', dest='use_reward_to_go', help='Use total trajectory reward (Formulation 1)')
    
    # For --use-advantage-norm
    parser.add_argument('--use-advantage-norm', action='store_true', default=True, help='Normalize advantages')
    parser.add_argument('--no-advantage-norm', action='store_false', dest='use_advantage_norm', help='Do NOT normalize advantages')
    
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Number of episodes to collect before one update')
    args = parser.parse_args()
    main(args)