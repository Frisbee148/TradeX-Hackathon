import torch
import numpy as np
import os
from .env import MarketEnv
from .overseer import Overseer, encode_observation, action_map

def run_evaluation(num_episodes=50, use_overseer=True, deterministic=True, pure_rule_based=False):
    env = MarketEnv()
    
    policy = None
    if use_overseer and not pure_rule_based:
        policy = Overseer()
        if os.path.exists("models/best_model.pth"):
            try:
                policy.load_state_dict(torch.load("models/best_model.pth", map_location="cpu", weights_only=True))
            except:
                pass
        policy.eval()

    final_prices = []
    rewards = []
    bots_blocked = []
    missed_attacks = []
    false_positives = []
    
    action_counts = {"ALLOW": 0, "BLOCK_NormalTrader": 0, "BLOCK_Manipulator": 0, "BLOCK_Arbitrage": 0, "BLOCK_NoisyTrader": 0}
    
    for ep in range(num_episodes):
        seed = 1000 + ep
        obs = env.reset(stage=5, seed=seed)
        total_reward = 0
        done = False
        
        while not done:
            if use_overseer:
                if pure_rule_based:
                    # Simple rule baseline
                    if obs.get("threat_score", 0) > 0.8:
                        action_str = "BLOCK_Manipulator"
                    else:
                        action_str = "ALLOW"
                else:
                    obs_vec = encode_observation(obs)
                    action_idx, _, _, _, _ = policy.select_action(obs_vec, deterministic=deterministic)
                    action_str = action_map[action_idx]
            else:
                action_str = "ALLOW"
                
            obs, reward, done, info = env.step(action_str)
            total_reward += reward
            
            if action_str == "ALLOW":
                action_counts["ALLOW"] += 1
            elif action_str.startswith("BLOCK_"):
                blocked_id = int(action_str.split("_")[1]) if "_" in action_str and action_str.split("_")[1].isdigit() else -1
                if blocked_id != -1:
                    target_agent = info['agent_types'].get(blocked_id, 'Unknown')
                    k = f"BLOCK_{target_agent}"
                    action_counts[k] = action_counts.get(k, 0) + 1
            
            if len(bots_blocked) <= ep:
                bots_blocked.append(0)
            bots_blocked[-1] += info.get("correct_detect", 0.0)
                
            if len(false_positives) <= ep:
                false_positives.append(0)
            false_positives[-1] += info.get("false_positive", 0.0)
                
            if len(missed_attacks) <= ep:
                missed_attacks.append(0)
            missed_attacks[-1] += info.get("missed_attack", 0.0)
                
        final_prices.append(info["final_price"])
        rewards.append(total_reward)

    total_actions = sum(action_counts.values())
    action_dist = {k: (v / total_actions) * 100.0 if total_actions > 0 else 0.0 for k, v in action_counts.items()}

    tp = sum(bots_blocked)
    fp = sum(false_positives)
    fn = sum(missed_attacks)
    
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    intervention_rate = ((total_actions - action_counts["ALLOW"]) / total_actions) * 100.0 if total_actions > 0 else 0.0
    
    return {
        "avg_final_price_error": np.mean([abs(p - 100.0) for p in final_prices]),
        "price_std": np.std(final_prices),
        "avg_reward": np.mean(rewards),
        "bots_blocked": np.mean(bots_blocked), 
        "missed_attacks": np.mean(missed_attacks),
        "false_positives": np.mean(false_positives),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "intervention_rate": intervention_rate,
        "action_dist": action_dist
    }

def main():
    print("\nEvaluations using models/best_model.pth\n")
    print("Evaluating baseline (No Overseer)...")
    no_overseer = run_evaluation(num_episodes=50, use_overseer=False)
    
    print("Evaluating trained model (Mode A: Deterministic)...")
    with_overseer_det = run_evaluation(num_episodes=50, use_overseer=True, deterministic=True)
    
    print("Evaluating trained model (Mode B: Stochastic Evaluation)...")
    with_overseer_stoch = run_evaluation(num_episodes=50, use_overseer=True, deterministic=False)
    
    print("Evaluating trained model (Mode C: Rule-Based Hybrid Baseline)...")
    with_overseer_rule = run_evaluation(num_episodes=50, use_overseer=True, pure_rule_based=True)
    
    print("\n" + "=" * 110)
    print("                                      HACKATHON BENCHMARK REPORT                ")
    print("=" * 110)
    print("                      | WITHOUT Overseer | WITH Overseer (Det) | WITH Overseer (Stoch) | WITH Overseer (Hybrid)")
    print("-" * 110)
    print(f"Avg Reward:           |        {no_overseer['avg_reward']:7.1f} |           {with_overseer_det['avg_reward']:7.1f} |             {with_overseer_stoch['avg_reward']:7.1f} |            {with_overseer_rule['avg_reward']:7.1f}")
    print(f"Price Error:          |        {no_overseer['avg_final_price_error']:7.2f} |           {with_overseer_det['avg_final_price_error']:7.2f} |             {with_overseer_stoch['avg_final_price_error']:7.2f} |            {with_overseer_rule['avg_final_price_error']:7.2f}")
    print(f"Volatility (Std):     |        {no_overseer['price_std']:7.2f} |           {with_overseer_det['price_std']:7.2f} |             {with_overseer_stoch['price_std']:7.2f} |            {with_overseer_rule['price_std']:7.2f}")
    print(f"Correct Blocks (TP):  |        {no_overseer['bots_blocked']:7.1f} |           {with_overseer_det['bots_blocked']:7.1f} |             {with_overseer_stoch['bots_blocked']:7.1f} |            {with_overseer_rule['bots_blocked']:7.1f}")
    print(f"Missed Attacks (FN):  |        {no_overseer['missed_attacks']:7.1f} |           {with_overseer_det['missed_attacks']:7.1f} |             {with_overseer_stoch['missed_attacks']:7.1f} |            {with_overseer_rule['missed_attacks']:7.1f}")
    print(f"False Positives (FP): |        {no_overseer['false_positives']:7.1f} |           {with_overseer_det['false_positives']:7.1f} |             {with_overseer_stoch['false_positives']:7.1f} |            {with_overseer_rule['false_positives']:7.1f}")
    print(f"Intervention Rate:    |          {0.0:5.1f}% |            {with_overseer_det['intervention_rate']:5.1f}% |              {with_overseer_stoch['intervention_rate']:5.1f}% |             {with_overseer_rule['intervention_rate']:5.1f}%")
    print(f"Precision:            |          {0.0:5.1f}% |            {with_overseer_det['precision']:5.1f}% |              {with_overseer_stoch['precision']:5.1f}% |             {with_overseer_rule['precision']:5.1f}%")
    print(f"Recall:               |          {0.0:5.1f}% |            {with_overseer_det['recall']:5.1f}% |              {with_overseer_stoch['recall']:5.1f}% |             {with_overseer_rule['recall']:5.1f}%")
    print(f"F1 Score:             |          {0.0:5.1f}% |            {with_overseer_det['f1_score']:5.1f}% |              {with_overseer_stoch['f1_score']:5.1f}% |             {with_overseer_rule['f1_score']:5.1f}%")
    print("-" * 110)
    
    loss_prev = with_overseer_stoch['avg_reward'] - no_overseer['avg_reward']
    stab_gain = ((no_overseer['avg_final_price_error'] - with_overseer_stoch['avg_final_price_error']) / max(0.1, no_overseer['avg_final_price_error'])) * 100
    print(f"Manipulator Loss Prevented (Stoch): +{loss_prev:.1f} Reward")
    print(f"Stability Gain (Stoch): {stab_gain:.1f}%")

    print("\nAction Distribution (Stochastic):")
    allow_pct = 0.0
    for k, v in with_overseer_stoch['action_dist'].items():
        print(f"{k:20s}: {v:5.1f}%")
        if k == "ALLOW": allow_pct = v
        
    if allow_pct > 90.0:
        print("\n> WARNING: Policy likely collapsed into passive ALLOW strategy.")
        print("> The model prefers silence because it missed block thresholds. Train longer!")

if __name__ == "__main__":
    main()