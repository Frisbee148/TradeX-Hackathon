import os
from typing import Dict, Generator, Tuple

import gradio as gr
import pandas as pd

from tradex.compare_all import compare_all
from tradex.eval_trl import (
    DEFAULT_TRL_PATH,
    DEFAULT_UNSLOTH_PATH,
    generate_trl_action,
    load_trl_model,
)
from tradex.env import MarketEnv
from tradex.text_adapter import observation_to_prompt, text_action_to_env_action


def _load_selected_model(model_choice: str):
    path = DEFAULT_TRL_PATH if model_choice == "TRL Overseer" else DEFAULT_UNSLOTH_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return load_trl_model(path), path


def run_live_replay(seed: int, stage: int, model_choice: str) -> Generator[Tuple[str, str, str, str, str, str], None, None]:
    (model, tokenizer), _ = _load_selected_model(model_choice)
    env = MarketEnv()
    obs = env.reset(seed=int(seed), stage=int(stage))
    done = False

    while not done:
        response, text_action, env_action = generate_trl_action(model, tokenizer, obs)
        next_obs, reward, done, info = env.step(env_action)

        state = (
            f"Price: {obs['price']:.2f}\n"
            f"Volatility: {obs['volatility']:.3f}\n"
            f"Liquidity: {obs['liquidity']:.2f}\n"
            f"Timestep: {obs['timestep']}"
        )
        threat = f"{info.get('threat_score', 0.0):.2f}"
        manip = "Active" if info.get("is_attack_active", False) else "Inactive"
        llm_resp = response.strip() or "(empty -> parsed as ALLOW)"
        mapped = env_action
        rew = f"{float(reward):.3f}"

        yield state, threat, manip, llm_resp, mapped, rew
        obs = next_obs


def load_reward_curves():
    return (
        "plots/trl_reward_vs_episode.png" if os.path.exists("plots/trl_reward_vs_episode.png") else None,
        "plots/trl_loss_curve.png" if os.path.exists("plots/trl_loss_curve.png") else None,
        "plots/trl_precision_recall.png" if os.path.exists("plots/trl_precision_recall.png") else None,
    )


def load_comparison_df(episodes: int):
    df, _ = compare_all(episodes=episodes, trl_model_path=DEFAULT_TRL_PATH, unsloth_model_path=DEFAULT_UNSLOTH_PATH)
    return df


def explain_one(seed: int, stage: int, step: int, model_choice: str):
    (model, tokenizer), model_path = _load_selected_model(model_choice)
    env = MarketEnv()
    obs = env.reset(seed=int(seed), stage=int(stage))

    for _ in range(max(0, int(step) - 1)):
        obs, _, done, _ = env.step("ALLOW")
        if done:
            break

    prompt = observation_to_prompt(obs)
    response, text_action, env_action = generate_trl_action(model, tokenizer, obs)

    why = (
        f"Threat score {obs.get('threat_score', 0.0):.2f} drove a `{text_action}` decision; "
        f"environment action mapped to `{env_action}` for runtime compatibility."
    )
    return prompt, response, f"{text_action} -> {env_action}", why, model_path


def run_attack_scenario(scenario: str, model_choice: str):
    (model, tokenizer), model_path = _load_selected_model(model_choice)
    env = MarketEnv()
    obs = env.reset(seed=42, stage=5)

    if scenario == "Pump & Dump":
        obs["threat_score"] = 0.92
    elif scenario == "Spoofing":
        obs["threat_score"] = 0.78
    elif scenario == "Burst Manipulation":
        obs["threat_score"] = 0.86
    elif scenario == "Sandwich-like Attack":
        obs["threat_score"] = 0.73
    else:
        obs["threat_score"] = 0.22

    response, text_action, env_action = generate_trl_action(model, tokenizer, obs)
    _, reward, _, info = env.step(text_action_to_env_action(text_action, obs))

    return (
        f"Scenario: {scenario}\nModel: {model_path}",
        f"LLM Response: {response.strip()}",
        f"Chosen Action: {text_action}",
        f"Mapped Env Action: {env_action}",
        f"Reward: {float(reward):.3f} | Threat: {info.get('threat_score', 0.0):.2f}",
    )


def build_demo():
    with gr.Blocks(theme=gr.themes.Base(), css="body { background: #0f1117; color: #e8eaed; } .gradio-container {max-width: 1400px !important;}") as demo:
        gr.Markdown("# TradeX TRL Governance Dashboard")
        gr.Markdown(
            "LLM-trained market overseer using Hugging Face TRL operating inside a multi-agent AMM adversarial simulation."
        )

        with gr.Tab("Live TRL Replay"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        choices=["TRL Overseer", "TRL Unsloth Overseer"],
                        value="TRL Overseer",
                        label="Model",
                    )
                    seed = gr.Number(value=42, label="Seed")
                    stage = gr.Slider(minimum=1, maximum=5, value=5, step=1, label="Stage")
                    run_btn = gr.Button("Run Streaming Replay", variant="primary")
                with gr.Column(scale=2):
                    market_state = gr.Textbox(label="Current Market State", lines=6)
                    threat = gr.Textbox(label="Threat Score")
                    manip = gr.Textbox(label="Manipulator Activity")
                    llm_response = gr.Textbox(label="LLM Generated Governance Response", lines=3)
                    mapped_action = gr.Textbox(label="Mapped Env Action")
                    reward = gr.Textbox(label="Reward Received")

            run_btn.click(
                fn=run_live_replay,
                inputs=[seed, stage, model_choice],
                outputs=[market_state, threat, manip, llm_response, mapped_action, reward],
            )

        with gr.Tab("TRL Reward Curves"):
            refresh_btn = gr.Button("Refresh Curves")
            reward_img = gr.Image(label="TRL Reward vs Episode")
            loss_img = gr.Image(label="TRL Loss Curve")
            pr_img = gr.Image(label="TRL Precision / Recall")
            refresh_btn.click(fn=load_reward_curves, outputs=[reward_img, loss_img, pr_img])

        with gr.Tab("PPO vs TRL Comparison"):
            cmp_episodes = gr.Slider(minimum=20, maximum=200, value=100, step=20, label="Episodes")
            cmp_btn = gr.Button("Load Benchmark", variant="primary")
            cmp_df = gr.Dataframe(label="Final Benchmark Table", wrap=True)
            cmp_btn.click(fn=load_comparison_df, inputs=cmp_episodes, outputs=cmp_df)

        with gr.Tab("Explainability"):
            with gr.Row():
                ex_model = gr.Dropdown(
                    choices=["TRL Overseer", "TRL Unsloth Overseer"],
                    value="TRL Overseer",
                    label="Model",
                )
                ex_seed = gr.Number(value=42, label="Seed")
                ex_stage = gr.Slider(minimum=1, maximum=5, value=5, step=1, label="Stage")
                ex_step = gr.Number(value=5, label="Step")
                ex_btn = gr.Button("Explain Decision", variant="primary")
            ex_prompt = gr.Textbox(label="Prompt Sent To Model", lines=10)
            ex_resp = gr.Textbox(label="Generated Response", lines=3)
            ex_action = gr.Textbox(label="Chosen Action")
            ex_why = gr.Textbox(label="Why Block/Allow")
            ex_model_path = gr.Textbox(label="Model Path")
            ex_btn.click(
                fn=explain_one,
                inputs=[ex_seed, ex_stage, ex_step, ex_model],
                outputs=[ex_prompt, ex_resp, ex_action, ex_why, ex_model_path],
            )

        with gr.Tab("Attack Scenarios"):
            sc_model = gr.Dropdown(
                choices=["TRL Overseer", "TRL Unsloth Overseer"],
                value="TRL Overseer",
                label="Model",
            )
            scenario = gr.Dropdown(
                choices=[
                    "Pump & Dump",
                    "Spoofing",
                    "Burst Manipulation",
                    "Sandwich-like Attack",
                    "Passive Normal Market",
                ],
                value="Pump & Dump",
                label="Scenario",
            )
            sc_btn = gr.Button("Run Scenario", variant="primary")
            sc_meta = gr.Textbox(label="Scenario Context", lines=2)
            sc_resp = gr.Textbox(label="Model Response", lines=2)
            sc_action = gr.Textbox(label="Chosen Action")
            sc_mapped = gr.Textbox(label="Mapped Action")
            sc_reward = gr.Textbox(label="Outcome")
            sc_btn.click(
                fn=run_attack_scenario,
                inputs=[scenario, sc_model],
                outputs=[sc_meta, sc_resp, sc_action, sc_mapped, sc_reward],
            )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
