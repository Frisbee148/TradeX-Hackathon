import gradio as gr
import torch
import numpy as np
import os
from tradex.env import MarketEnv
from tradex.overseer import Overseer, encode_observation, action_map

policy = Overseer()
if os.path.exists("models/best_model.pth"):
    try:
        policy.load_state_dict(torch.load("models/best_model.pth", map_location="cpu", weights_only=True))
    except:
        pass
policy.eval()

def run_single_episode(seed, stage, use_overseer):
    env = MarketEnv()
    obs = env.reset(stage=int(stage), seed=int(seed))
    
    logs = []
    total_reward = 0
    allow_count = 0
    correct_blocks = 0
    missed_attacks = 0
    false_positives = 0
    action_counts = {"ALLOW": 0, "BLOCK_Manipulator": 0, "BLOCK_NormalTrader": 0, "BLOCK_Arbitrage": 0, "BLOCK_NoisyTrader": 0}
    
    done = False
    max_threat = 0
    
    while not done:
        if use_overseer:
            obs_vec = encode_observation(obs)
            action_idx, _, _, _, probs = policy.select_action(obs_vec, deterministic=True)
            action_str = action_map[action_idx]
            if action_str == "ALLOW": allow_count += 1
        else:
            action_str = "ALLOW"
            
        old_price = env.price
        obs, reward, done, info = env.step(action_str)
        total_reward += reward
        correct_blocks += info.get("correct_detect", 0)
        missed_attacks += info.get("missed_attack", 0)
        false_positives += info.get("false_positive", 0)
        
        target_str = ""
        if action_str != "ALLOW" and use_overseer:
            blocked_id = int(action_str.split("_")[1]) if "_" in action_str and action_str.split("_")[1].isdigit() else -1
            t_agent = info['agent_types'].get(blocked_id, 'Unknown')
            target_str = f"BLOCK_{t_agent}"
            if target_str in action_counts:
                action_counts[target_str] += 1
        elif action_str == "ALLOW":
            action_counts["ALLOW"] += 1
            
        threat = info["threat_score"]
        if threat > max_threat: max_threat = threat
        
        step_log = "-" * 50 + f"\nStep {env.timestep:02d}\nPrice: {old_price:.1f}\n\nActions:\n"
        agents_acted = {t['agent']: t for t in info["executed_trades"]}
        if "intended_trades" in info and action_str != "ALLOW":
            blocked_id = int(action_str.split("_")[1]) if "_" in action_str and action_str.split("_")[1].isdigit() else -1
            blocked_intent = [t for t in info["intended_trades"] if t['agent'] == blocked_id]
            if blocked_intent:
                agents_acted[blocked_id] = blocked_intent[0]
                
        for i, a_type in info['agent_types'].items():
            if i in agents_acted:
                t = agents_acted[i]
                step_log += f"{a_type:14s} -> {t['action']} {t['size']:.1f}\n"
            else:
                step_log += f"{a_type:14s} -> HOLD\n"
            
        step_log += f"\nOverseer Analysis:\nThreat Score: {threat:.2f}\nDetected:\n"
        if threat > 0.3:
            lines = info["block_reason"].split("- ")
            for line in lines:
                if line.strip():
                    step_log += f"- {line.strip()}\n"
        else:
            step_log += "- routine liquidity flow\n"
            
        if use_overseer and target_str:
            confidence = probs[action_idx].item() * 100
            step_log += f"\nDecision:\n{target_str}\n\nConfidence:\n{confidence:.0f}%\n"
            step_log += f"\nOutcome:\nTrade cancelled\nPrice stabilized to {env.price:.1f}\nReward: {reward:.2f}\n"
        else:
            step_log += f"\nDecision:\nALLOW\n"
            if use_overseer:
                confidence = probs[0].item() * 100
                step_log += f"\nConfidence:\n{confidence:.0f}%\n"
            step_log += f"\nOutcome:\nTrade allowed\nPrice becomes {env.price:.1f}\nReward: {reward:.2f}\n"

        logs.append(step_log)
        
    intervention_rate = ((env.max_steps - allow_count) / env.max_steps) * 100 if use_overseer else 0.0
    
    tp = correct_blocks
    fp = false_positives
    fn = missed_attacks
    precision = (tp / (tp+fp))*100 if (tp+fp)>0 else 0
    recall = (tp / (tp+fn))*100 if (tp+fn)>0 else 0
    stability_gain = 100 - (abs((env.price - 100.0) / 100.0) * 100)
    
    threat_level = "CRITICAL" if max_threat > 0.85 else "ELEVATED" if max_threat > 0.5 else "SAFE"
    
    return [
        f"**Threat Level:** {threat_level} ({max_threat:.2f})",
        f"**Intervention Rate:** {intervention_rate:.1f}%",
        f"**Correct Blocks:** {correct_blocks}",
        f"**Missed Attacks:** {missed_attacks}",
        f"**Precision:** {precision:.1f}%",
        f"**Recall:** {recall:.1f}%",
        f"**Stability Gain:** {stability_gain:.1f}%",
        "\n".join(logs)
    ]

def load_plot(plot_name):
    path = f"plots/{plot_name}"
    if os.path.exists(path):
        return path
    return None

with gr.Blocks(theme=gr.themes.Monochrome(primary_hue="slate", neutral_hue="slate")) as demo:
    gr.Markdown("# 📉 TradeX — DeepMind Style AI Governance Control Room")
    gr.Markdown("A hybrid rule-based anomaly detector passes high-fidelity threat scores context into the Deep PPO architecture to govern autonomous agents.")
    
    with gr.Tab("Live Market Oversight Replay"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Dashboard Controls")
                seed_input = gr.Number(value=4021, label="Simulation Seed")
                stage_input = gr.Slider(minimum=1, maximum=5, value=5, step=1, label="Market Chaos Stage")
                
                use_ai = gr.Checkbox(value=True, label="Enable Deep RL Overseer")
                run_btn = gr.Button("Execute Market Simulation", variant="primary")
                
                gr.Markdown("### Episode Diagnostics")
                out_threat = gr.Markdown()
                out_int = gr.Markdown()
                out_cb = gr.Markdown()
                out_ma = gr.Markdown()
                out_prec = gr.Markdown()
                out_rec = gr.Markdown()
                out_stab = gr.Markdown()
            
            with gr.Column(scale=2):
                ep_logs = gr.Textbox(label="Autonomous Agent & Model Reasoning", lines=28, max_lines=40)
                
        run_btn.click(
            run_single_episode, 
            inputs=[seed_input, stage_input, use_ai], 
            outputs=[out_threat, out_int, out_cb, out_ma, out_prec, out_rec, out_stab, ep_logs]
        )
        
    with gr.Tab("Reinforcement Learning PPO Data"):
        
        with gr.Row():
            p1 = gr.Image(value=load_plot("reward_vs_episode.png"), label="Reward Moving Average")
            p2 = gr.Image(value=load_plot("precision_recall.png"), label="Precision / Recall Boundary")
            
        with gr.Row():
            p3 = gr.Image(value=load_plot("detection_capability.png"), label="Correct Blocks vs False Positives")
            p4 = gr.Image(value=load_plot("final_price_error_vs_episode.png"), label="Action Distribution Target")

if __name__ == "__main__":
    demo.launch()
