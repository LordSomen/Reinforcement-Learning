import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import argparse
import time

# --- Define the Network Architecture ---
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
        return Categorical(logits=logits)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Watch a trained Policy Gradient agent.')
    
    # First argument: model_path
    parser.add_argument('--model_path', type=str,
                        help='Path to the saved model state_dict (.pth file)')
    
    # Second argument: env_name
    parser.add_argument('--env_name', type=str,
                        help='Name of the Gym environment (e.g., CartPole-v1)')
    
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the environment with rendering
    env = gym.make(args.env_name, render_mode="human")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = PolicyNetwork(state_size, action_size).to(device)
    
    try:
        # Load the saved weights
        policy_net.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        env.close()
        exit()
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Ensure the network architecture in this script matches the saved model.")
        env.close()
        exit()

    # Set the network to evaluation mode 
    policy_net.eval()

    print(f"--- 🚀 Watching trained agent in {args.env_name} ---")
    
    # --- Run One Full Episode ---
    state, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0

    while not (terminated or truncated):
        # Convert state to a tensor
        state_t = torch.tensor(state, dtype=torch.float32).to(device)

        with torch.no_grad():
            # Get the action distribution
            action_dist = policy_net(state_t)

            # We use .logits.argmax() to pick the action with the
            # highest probability (the most "confident" action).
            action = action_dist.logits.argmax().item()
        
        # Take the best action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        state = next_state

        # time.sleep(0.01)

    print(f" Episode finished. Total Reward: {total_reward} ---")
    env.close()