import torch
import numpy as np
from .env import MarketEnv
from .train import SimplePolicy, encode_observation

def run_episodes(policy, num_episodes=50, always_allow=False):
    env = MarketEnv()
    final_prices = []
    rewards = []
    bots_blocked = []
    false_positives = []

    for _ in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            if always_allow:
                action = "ALLOW"
            else:
                obs_vec = encode_observation(obs)
                action, _ = policy.select_action(obs_vec)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        final_prices.append(info["final_price"])
        rewards.append(total_reward)
        bots_blocked.append(info["bots_blocked"])
        false_positives.append(info["false_positives"])

    return {
        "avg_final_price": np.mean(final_prices),
        "price_std": np.std(final_prices),
        "avg_reward": np.mean(rewards),
        "avg_bots_blocked": np.mean(bots_blocked),
        "avg_false_positives": np.mean(false_positives)
    }

def main():
    # Load trained policy
    policy = SimplePolicy()
    policy.load_state_dict(torch.load("policy.pth"))
    policy.eval()

    # Run without overseer
    no_overseer = run_episodes(policy, always_allow=True)

    # Run with overseer
    with_overseer = run_episodes(policy, always_allow=False)

    # Print comparison
    print("=" * 60)
    print("           WITHOUT Overseer    WITH Overseer")
    print("-" * 60)
    print(f"Avg Final Price:    {no_overseer['avg_final_price']:6.1f}           {with_overseer['avg_final_price']:6.1f}")
    print(f"Price Std Dev:      {no_overseer['price_std']:6.1f}           {with_overseer['price_std']:6.1f}")
    print(f"Avg Reward:         {no_overseer['avg_reward']:6.1f}           {with_overseer['avg_reward']:6.1f}")
    print(f"Bots Blocked/ep:    {no_overseer['avg_bots_blocked']:6.1f}           {with_overseer['avg_bots_blocked']:6.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()