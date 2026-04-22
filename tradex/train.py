import argparse
import random
import os
import torch
import torch.nn as nn
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

def pretrain_supervised(policy, env, optimizer, device):
    print("Collecting synthetic states for hierarchical rule-engine pretraining...")
    pretrain_obs = []
    pretrain_targets = []
    
    # 250 episodes * 50 steps = 12.5k synthetic states
    for ep in range(250):
        o = env.reset(stage=1, seed=ep)
        d = False
        while not d:
            o_vec = encode_observation(o)
            threat = o.get("threat_score", 0.0)
            
            if threat > 0.85:
                manip_idx = next(i for i, a in enumerate(env.agents) if getattr(a, 'is_malicious', False))
                target_action = manip_idx + 1 # BLOCK_Manipulator
            else:
                target_action = 0 # ALLOW
                
            pretrain_obs.append(o_vec)
            pretrain_targets.append(target_action)
            o, _, d, _ = env.step("ALLOW") 
            
    print("Pretraining hierarchical architecture on rules...")
    p_obs_t = torch.tensor(np.array(pretrain_obs), dtype=torch.float32).to(device)
    p_targ_t = torch.tensor(pretrain_targets, dtype=torch.int64).to(device)
    p_dataset = TensorDataset(p_obs_t, p_targ_t)
    p_loader = DataLoader(p_dataset, batch_size=64, shuffle=True)
    
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(8):
        for b_o, b_t in p_loader:
            optimizer.zero_grad()
            probs, _ = policy(b_o)
            logits = torch.log(probs + 1e-8)
            loss = loss_fn(logits, b_t)
            loss.backward()
            optimizer.step()
            
    print("Pretraining complete.")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(args, 'onsite', False) else "cpu")
    print(f"Training on device: {device}")
    
    env = MarketEnv()
    policy = Overseer().to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    start_ep = 0
    if os.path.exists("models/best_model.pth"):
        try:
            policy.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=True))
        except RuntimeError as e:
            print("Architecture change detected. Starting training from scratch (Discarding old model).")
            pretrain_supervised(policy, env, optimizer, device)
    else:
        pretrain_supervised(policy, env, optimizer, device)
        
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    
    metrics_history = []
    best_reward = -float('inf')
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    clip_epsilon = 0.2
    ppo_epochs = 4
    mini_batch_size = 64
    
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
        
        allow_streak = getattr(args, 'allow_streak', 0)
        
        while not done:
            obs_vec = encode_observation(obs)
            x_tensor = torch.tensor(obs_vec, dtype=torch.float32).to(device).unsqueeze(0)
            threat = obs.get("threat_score", 0.0)
            
            with torch.no_grad():
                probs, val = policy(x_tensor)
                
                # Mask ALLOW if threat is absolute critical > 0.90
                if threat > 0.90:
                    probs[0, 0] = 0.0
                    probs = probs / probs.sum()
                    
                # 30% forced exploration into Intervene branch for first 300 episodes
                if episode < 300 and random.random() < 0.30:
                    probs[0, 0] *= 0.1
                    probs = probs / probs.sum()
                    
                dist = torch.distributions.Categorical(probs)
                action_tensor = dist.sample()
                action_idx = action_tensor.item()
                log_prob = dist.log_prob(action_tensor)
                confidence = probs[0][action_idx].item() * 100
                
            action_str = action_map[action_idx]
            
            # Record who attacked and who was blocked for strict logging
            malicious_agent_type = "Manipulator" if obs.get("malicious_active", False) else "None"
            
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
            
            if verbose_step and (threat > 0.8 or action_str != "ALLOW"):
                print("-" * 50)
                print(f"Step {env.timestep} | Threat Score: {threat:.2f}")
                print(f"Targeting logic:")
                print(f"  Chosen action: {action_str}")
                print(f"  Confidence: {confidence:.0f}%")
                
                if info["correct_detect"] > 0:
                    print("  Result: [TRUE POSITIVE] +3.0 Reward")
                elif info["false_positive"] > 0:
                    print("  Result: [FALSE POSITIVE] -0.7 Penalty")
                elif info["missed_attack"] > 0:
                    print("  Result: [FALSE NEGATIVE] -3.0 Penalty")
                else:
                    print("  Result: [TRUE NEGATIVE] +0.05 Reward")
                    
                print(f"  Reasoning: {info['block_reason'] if action_str != 'ALLOW' else 'Allowed flow'}")

            step_logs.append({
                "timestep": env.timestep,
                "action": action_str,
                "reward": float(reward)
            })
            
        total_actions = sum(action_counts.values())
        allow_pct = (action_counts["ALLOW"] / total_actions) * 100 if total_actions > 0 else 100.0
        
        # Collapse Recovery System
        if allow_pct > 95:
            args.allow_streak = getattr(args, 'allow_streak', 0) + 1
            if args.allow_streak >= 50:
                print(">>> ALLOW COLLAPSE DETECTED. Reloading last good checkpoint...")
                if os.path.exists("models/best_model.pth"):
                    policy.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=True))
                optimizer = optim.Adam(policy.parameters(), lr=1e-3)
                args.allow_streak = 0
        else:
            args.allow_streak = 0
        
        target_entropy = max(0.02, 0.05 * (1 - (episode / args.episodes)))
        
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
                probs, b_vals = policy(b_obs)
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
            
            # Save best model by: 3*Recall + 2*Precision + Reward
            val_metric = (3 * recall) + (2 * precision) + avg_rew
            
            log_str = f"\n==================================================\n"
            log_str += f"Episode {episode+1} | Stage {stage} | Seed {seed}\n"
            log_str += f"--------------------------------------------------\n"
            log_str += f"Training Metrics:\n"
            log_str += f"Total Reward: {ep_total_reward:.2f}\n"
            log_str += f"Rolling Avg Reward: {avg_rew:.2f}\n"
            log_str += f"Policy Loss: {p_loss:.3f}\n"
            log_str += f"Value Loss: {v_loss:.3f}\n"
            log_str += f"Entropy: {target_entropy:.4f}\n"
            log_str += f"Validation Score (3*Rec + 2*Prec + Rew): {val_metric:.1f}\n\n"
            
            log_str += f"Intervention Stats:\n"
            log_str += f"True Positives (Correct Blocks): {tp}\n"
            log_str += f"False Negatives (Missed Attacks): {fn}\n"
            log_str += f"False Positives: {fp}\n"
            log_str += f"Precision: {precision:.0f}%\n"
            log_str += f"Recall: {recall:.0f}%\n"
            log_str += f"Intervention Rate: {100.0 - allow_pct:.0f}%\n\n"
            
            log_str += f"Policy Distribution:\n"
            for k, v in action_counts.items():
                pct = (v / total_actions) * 100.0 if total_actions > 0 else 0
                log_str += f"{k}: {pct:.0f}%\n"
            log_str += f"==================================================\n"
            
            print(log_str)
            
            if val_metric > best_reward and recall > 0 and (100.0 - allow_pct) >= 5.0:
                best_reward = val_metric
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