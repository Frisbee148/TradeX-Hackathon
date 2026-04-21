import matplotlib.pyplot as plt
import numpy as np

def log_step(episode, timestep, action, reward, price, suspicion_scores):
    print(f"[Ep {episode:2d} | Step {timestep:2d}] Action: {action} | Reward: {reward:+.1f} | Price: {price:.1f} | Suspicion: {suspicion_scores}")

def plot_reward_curve(reward_history):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Episode Reward')
    if len(reward_history) >= 10:
        rolling_avg = np.convolve(reward_history, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(reward_history)), rolling_avg, label='Rolling Average (10)', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Overseer Learning Curve — TradeX')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_curve.png')
    plt.show()

def plot_price_stability(price_history_no_overseer, price_history_with_overseer):
    plt.figure(figsize=(10, 5))
    plt.plot(price_history_no_overseer, label='Without Overseer', alpha=0.7)
    plt.plot(price_history_with_overseer, label='With Overseer', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Final Price')
    plt.title('Price Stability Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('price_stability.png')
    plt.show()