import argparse
import random
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json

from .env import MarketEnv
from .overseer import Overseer, encode_observation, action_map
from .utils import plot_all_metrics, save_episode_log

def get_stage(episode):
    if episode <= 300: return 1
    elif episode <= 700: return 2
    elif episode <= 1200: return 3
    elif episode <= 1800: return 4
    else: return 5

def compute_gae(rewards, values, gamma=0.99, tau=0.95):
    gae = 0
    returns = []
    values = values + [0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * tau * gae
        returns.insert(0, gae + values[step])
    return returns

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(args, 'onsite', False) else "cpu")
    print(f"Training on device: {device}")
    
    env = MarketEnv()
    policy = Overseer().to(device)
    
    start_ep = 0
    if os.path.exists("models/best_model.pth"):
        try:
            policy.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=True))
        except RuntimeError as e:
            print("Architecture change detected. Starting training from scratch (Discarding old 53-dim model).")
        
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    
    metrics_history = []
    best_reward = -float('inf')
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    clip_epsilon = 0.2
    ppo_epochs = 4
    mini_batch_size = 64
    
    target_entropy = 0.10
    allow_streak = 0
    
    for episode in range(start_ep, args.episodes):
        stage = get_stage(episode)
        seed = 42 + episode
        obs = env.reset(stage=stage, seed=seed)
        
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        val_buffer = []
        reward_buffer = []
        
        ep_false_positives = 0
        ep_correct_blocks = 0
        ep_missed_attacks = 0
        ep_total_reward = 0.0
        step_logs = []
        done = False
        
        verbose_step = args.verbose or (episode > 0 and episode % 50 == 0)
        
        action_counts = {"ALLOW": 0, "BLOCK_NormalTrader": 0, "BLOCK_Manipulator": 0, "BLOCK_Arbitrage": 0, "BLOCK_NoisyTrader": 0}
        
        while not done:
            obs_vec = encode_observation(obs)
            x_tensor = torch.tensor(obs_vec, dtype=torch.float32).to(device).unsqueeze(0)
            
            with torch.no_grad():
                logits, val = policy(x_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                action_tensor = dist.sample()
                action_idx = action_tensor.item()
                log_prob = dist.log_prob(action_tensor)
                confidence = probs[0][action_idx].item() * 100
                
            action_str = action_map[action_idx]
            old_price = env.price
            
            obs, reward, done, info = env.step(action_str)
            
            ep_false_positives += info["false_positive"]
            ep_correct_blocks += info["correct_detect"]
            ep_missed_attacks += info["missed_attack"]
            ep_total_reward += reward
            
            if action_str == "ALLOW":
                action_counts["ALLOW"] += 1
            else:
                target_agent = info['agent_types'].get(action_idx-1, 'Unknown')
                action_counts[f"BLOCK_{target_agent}"] = action_counts.get(f"BLOCK_{target_agent}", 0) + 1
            
            obs_buffer.append(obs_vec)
            action_buffer.append(action_idx)
            log_prob_buffer.append(log_prob.item())
            val_buffer.append(val.item())
            reward_buffer.append(reward)
            
            if verbose_step:
                print("-" * 50)
                print(f"Step {env.timestep}")
                print(f"Price: {old_price:.1f}\n")
                print("Actions:")
                
                agents_acted = {t['agent']: t for t in info["executed_trades"] + ([t for t in info["intended_trades"] if t['agent'] == (action_idx-1)])}
                for i, a_type in info['agent_types'].items():
                    if i in agents_acted:
                        t = agents_acted[i]
                        print(f"{a_type:14s} -> {t['action']} {t['size']:.1f}")
                    else:
                        print(f"{a_type:14s} -> HOLD")
                        
                threat = info["threat_score"]
                print(f"\nOverseer Analysis:\nThreat Score: {threat:.2f}")
                
                if threat > 0.3:
                    print("Detected:")
                    lines = info["block_reason"].split("- ")
                    for line in lines:
                        if line.strip():
                            print(f"- {line.strip()}")
                else:
                    print("Detected:\n- routine liquidity flow")
                
                if action_str != "ALLOW":
                    real_target = f"BLOCK_{info['agent_types'].get(action_idx-1, 'Unknown')}"
                    print(f"\nDecision:\n{real_target}")
                    print(f"\nConfidence:\n{confidence:.0f}%")
                    print(f"\nOutcome:\nTrade cancelled\nPrice stabilized to {env.price:.1f}\nReward: {reward:.2f}")
                else:
                    print("\nDecision:\nALLOW")
                    print(f"\nConfidence:\n{confidence:.0f}%")
                    if threat > 0.5:
                        print(f"\nOutcome:\nCatastrophic miss!\nPrice crashes to {env.price:.1f}\nReward: {reward:.2f}")
                    else:
                        print(f"\nOutcome:\nNormal trading continued\nReward: {reward:.2f}")

            step_logs.append({
                "timestep": env.timestep,
                "price": float(env.price),
                "action": action_str,
                "reward": float(reward)
            })
            
        # Minimum intervention penalty
        if ep_correct_blocks == 0 and ep_false_positives == 0 and ep_missed_attacks > 0:
            reward_buffer[-1] -= 5.0 # Massive episode penalty
            ep_total_reward -= 5.0
            
        total_actions = sum(action_counts.values())
        allow_pct = (action_counts["ALLOW"] / total_actions) * 100 if total_actions > 0 else 100.0
        
        if allow_pct > 90:
            allow_streak += 1
            if allow_streak >= 100:
                target_entropy = min(0.6, target_entropy + 0.15)
                allow_streak = 0
        else:
            allow_streak = 0
            if target_entropy > 0.05:
                target_entropy *= 0.98
                
        returns = compute_gae(reward_buffer, val_buffer)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.tensor(val_buffer, dtype=torch.float32).to(device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        old_obs_t = torch.tensor(np.array(obs_buffer), dtype=torch.float32).to(device)
        old_actions_t = torch.tensor(action_buffer, dtype=torch.int64).to(device)
        old_log_probs_t = torch.tensor(log_prob_buffer, dtype=torch.float32).to(device)
        
        dataset = TensorDataset(old_obs_t, old_actions_t, old_log_probs_t, returns, advantages)
        loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
        
        policy_loss_sum = 0
        value_loss_sum = 0
        
        for _ in range(ppo_epochs):
            for b_obs, b_actions, b_old_log_probs, b_returns, b_advs in loader:
                logits, b_vals = policy(b_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_advs
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(b_vals.squeeze(-1), b_returns)
                
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                
                loss = policy_loss + 0.5 * value_loss - target_entropy * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()
                
        scheduler.step()
        
        tp = ep_correct_blocks
        fp = ep_false_positives
        fn = ep_missed_attacks
        precision = (tp / (tp+fp))*100 if (tp+fp)>0 else 0
        recall = (tp / (tp+fn))*100 if (tp+fn)>0 else 0
        
        ep_metrics = {
            "episode": episode,
            "seed": seed,
            "stage": stage,
            "reward": float(ep_total_reward),
            "false_positives": ep_false_positives,
            "bots_blocked": ep_correct_blocks,
            "price_error": info["price_error"]
        }
        metrics_history.append(ep_metrics)
            
        if (episode + 1) % 10 == 0:
            avg_rew = np.mean([m['reward'] for m in metrics_history[-10:]])
            p_loss = policy_loss_sum / (ppo_epochs * len(loader))
            v_loss = value_loss_sum / (ppo_epochs * len(loader))
            
            log_str = f"\n==================================================\n"
            log_str += f"Episode {episode+1} | Stage {stage} | Seed {seed}\n"
            log_str += f"--------------------------------------------------\n"
            log_str += f"Training Metrics:\n"
            log_str += f"Total Reward: {ep_total_reward:.2f}\n"
            log_str += f"Rolling Avg Reward: {avg_rew:.2f}\n"
            log_str += f"Policy Loss: {p_loss:.3f}\n"
            log_str += f"Value Loss: {v_loss:.3f}\n"
            log_str += f"Entropy: {target_entropy:.2f}\n"
            log_str += f"Learning Rate: {scheduler.get_last_lr()[0]:.5f}\n\n"
            
            log_str += f"Intervention Stats:\n"
            log_str += f"Correct Blocks: {ep_correct_blocks}\n"
            log_str += f"Missed Attacks: {ep_missed_attacks}\n"
            log_str += f"False Positives: {ep_false_positives}\n"
            log_str += f"Precision: {precision:.0f}%\n"
            log_str += f"Recall: {recall:.0f}%\n"
            log_str += f"Intervention Rate: {100.0 - allow_pct:.0f}%\n\n"
            
            log_str += f"Policy Distribution:\n"
            for k, v in action_counts.items():
                pct = (v / total_actions) * 100.0 if total_actions > 0 else 0
                log_str += f"{k}: {pct:.0f}%\n"
                
            vol_str = "High" if obs['volatility'] > 1.5 else "Low"
            log_str += f"\nEnvironment Summary:\n"
            log_str += f"Final Price: {env.price:.1f}\n"
            log_str += f"Price Error: {info['price_error']:.2f}\n"
            log_str += f"Volatility: {vol_str}\n"
            log_str += f"Liquidity: {info['total_liquidity']:.0f}\n"
            log_str += f"==================================================\n"
            
            print(log_str)
            
            if avg_rew > best_reward:
                best_reward = avg_rew
                torch.save(policy.state_dict(), "models/best_model.pth")
                
        save_episode_log(episode, seed, {"metrics": ep_metrics, "steps": step_logs})
        
    with open("metrics/training_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
        
    plot_all_metrics(metrics_history)
    print("\nTraining Complete! Plots saved to /plots. Best model in /models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--onsite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.onsite:
        args.episodes = max(args.episodes, 5000)
    train(args)