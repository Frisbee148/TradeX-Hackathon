import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .env import MarketEnv
from .overseer import Overseer
from .utils import log_step

class SimplePolicy(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, output_dim=5):
        super(SimplePolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = self.forward(obs_tensor)
        action_idx = torch.multinomial(probs, 1).item()
        actions = ["ALLOW", "BLOCK_0", "BLOCK_1", "BLOCK_2", "BLOCK_3"]
        return actions[action_idx], probs[0][action_idx]

def encode_observation(obs):
    # Flatten to 17-dim vector
    price = obs["price"]
    price_history = obs["price_history"][-10:] + [100.0] * (10 - len(obs["price_history"]))
    price_deviation = obs["price_deviation"]
    # Per-agent trade volume last 5 steps
    volumes = [0.0] * 4
    for trade in obs["trade_history"]:
        volumes[trade["agent_id"]] += trade["size"]
    timestep_norm = obs["timestep"] / 50.0  # Assuming num_steps=50
    return np.array([price] + price_history + [price_deviation] + volumes + [timestep_norm])

def train():
    env = MarketEnv()
    policy = SimplePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    reward_history = []

    for episode in range(200):
        obs = env.reset()
        total_reward = 0
        log_probs = []
        rewards = []
        done = False
        while not done:
            obs_vec = encode_observation(obs)
            action, log_prob = policy.select_action(obs_vec)
            obs, reward, done, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

        # Compute returns and update
        returns = []
        G = 0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode:3d} | Total Reward: {total_reward:+.1f} | "
                  f"Bots Blocked: {info['bots_blocked']} | "
                  f"False Positives: {info['false_positives']} | "
                  f"Final Price: {info['final_price']:.1f}")

    return policy, reward_history

if __name__ == "__main__":
    trained_policy, reward_history = train()
    torch.save(trained_policy.state_dict(), "policy.pth")