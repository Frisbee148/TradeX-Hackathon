import argparse
import csv
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compare import run_evaluation
from .env import MarketEnv
from .text_adapter import observation_to_prompt, parse_model_action, text_action_to_env_action


DEFAULT_TRL_PATH = "models/trl_overseer/final"
DEFAULT_UNSLOTH_PATH = "models/trl_overseer_unsloth/final"


def load_trl_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_trl_action(model, tokenizer, obs: Dict) -> Tuple[str, str, str]:
    prompt = observation_to_prompt(obs)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        full_ids = model.generate(
            input_ids,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_ids = full_ids[:, input_ids.shape[1] :]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    text_action = parse_model_action(response)
    env_action = text_action_to_env_action(text_action, obs)
    return response, text_action, env_action


def _heuristic_env_action(obs: Dict) -> Tuple[str, str]:
    threat = float(obs.get("threat_score", 0.0))
    if threat > 0.8:
        return "BLOCK", text_action_to_env_action("BLOCK", obs)
    if threat > 0.6:
        return "FLAG", "ALLOW"
    if threat > 0.4:
        return "MONITOR", "ALLOW"
    return "ALLOW", "ALLOW"


def evaluate_policy_episodes(
    policy_name: str,
    action_fn: Callable[[Dict], Tuple[str, str]],
    num_episodes: int = 100,
    seed_start: int = 3000,
) -> List[Dict]:
    env = MarketEnv()
    rows: List[Dict] = []

    for episode in range(num_episodes):
        obs = env.reset(stage=5, seed=seed_start + episode)
        done = False

        ep_reward = 0.0
        tp = fp = fn = 0
        allow_count = block_count = targeted_blocks = correct_targeted_blocks = 0
        price_errors = []
        price_series = [float(obs.get("price", 100.0))]

        while not done:
            text_action, env_action = action_fn(obs)
            next_obs, reward, done, info = env.step(env_action)

            ep_reward += float(reward)
            tp += int(info.get("correct_detect", 0))
            fp += int(info.get("false_positive", 0))
            fn += int(info.get("missed_attack", 0))
            price_errors.append(float(info.get("price_error", 0.0)))
            price_series.append(float(next_obs.get("price", 100.0)))

            if env_action == "ALLOW":
                allow_count += 1
            elif env_action.startswith("BLOCK_"):
                block_count += 1
                targeted_blocks += 1
                blocked_id = int(env_action.split("_")[1]) if env_action.split("_")[1].isdigit() else -1
                if blocked_id != -1 and blocked_id in info.get("malicious_ids", []):
                    correct_targeted_blocks += 1

            obs = next_obs

        total_steps = max(1, allow_count + block_count)
        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        targeted_acc = (correct_targeted_blocks / targeted_blocks) * 100.0 if targeted_blocks > 0 else 0.0

        rows.append(
            {
                "policy": policy_name,
                "episode": episode,
                "reward": ep_reward,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "intervention_rate": (block_count / total_steps) * 100.0,
                "allow_rate": (allow_count / total_steps) * 100.0,
                "block_rate": (block_count / total_steps) * 100.0,
                "targeted_block_accuracy": targeted_acc,
                "avg_price_error": float(np.mean(price_errors)) if price_errors else 0.0,
                "volatility": float(np.std(price_series)),
            }
        )

    return rows


def summarize_episode_rows(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    keys = [
        "reward",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "intervention_rate",
        "allow_rate",
        "block_rate",
        "targeted_block_accuracy",
        "avg_price_error",
        "volatility",
    ]
    summary = {"policy": rows[0]["policy"]}
    for k in keys:
        summary[k] = float(np.mean([float(r[k]) for r in rows]))
    return summary


def _model_action_fn(model, tokenizer):
    def _inner(obs: Dict) -> Tuple[str, str]:
        _, text_action, env_action = generate_trl_action(model, tokenizer, obs)
        return text_action, env_action

    return _inner


def _save_rows_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model_path(model_path: str, policy_name: str, episodes: int) -> Tuple[List[Dict], Dict]:
    model, tokenizer = load_trl_model(model_path)
    rows = evaluate_policy_episodes(policy_name, _model_action_fn(model, tokenizer), num_episodes=episodes)
    return rows, summarize_episode_rows(rows)


def main(args):
    all_rows: List[Dict] = []
    report_rows: List[Dict] = []

    # Heuristic baseline and PPO metrics from existing evaluation utility.
    heur = run_evaluation(num_episodes=args.episodes, use_overseer=True, pure_rule_based=True)
    ppo = run_evaluation(num_episodes=args.episodes, use_overseer=True, deterministic=True)

    # Heuristic per-episode rows for CSV completeness.
    heuristic_rows = evaluate_policy_episodes("Heuristic Baseline", _heuristic_env_action, num_episodes=args.episodes)
    all_rows.extend(heuristic_rows)

    report_rows.append(
        {
            "policy": "Heuristic Baseline",
            "avg_reward": float(heur["avg_reward"]),
            "precision": float(heur["precision"]),
            "recall": float(heur["recall"]),
            "f1": float(heur["f1_score"]),
        }
    )
    report_rows.append(
        {
            "policy": "PPO Overseer",
            "avg_reward": float(ppo["avg_reward"]),
            "precision": float(ppo["precision"]),
            "recall": float(ppo["recall"]),
            "f1": float(ppo["f1_score"]),
        }
    )

    trl_path = args.model_path
    unsloth_path = args.unsloth_model_path

    if os.path.exists(trl_path):
        trl_rows, trl_summary = evaluate_model_path(trl_path, "TRL Overseer", args.episodes)
        all_rows.extend(trl_rows)
        report_rows.append(
            {
                "policy": "TRL Overseer",
                "avg_reward": trl_summary["reward"],
                "precision": trl_summary["precision"],
                "recall": trl_summary["recall"],
                "f1": trl_summary["f1"],
            }
        )
    else:
        print(f"[WARN] TRL model not found at: {trl_path}")

    if os.path.exists(unsloth_path):
        uns_rows, uns_summary = evaluate_model_path(unsloth_path, "TRL Unsloth Overseer", args.episodes)
        all_rows.extend(uns_rows)
        report_rows.append(
            {
                "policy": "TRL Unsloth Overseer",
                "avg_reward": uns_summary["reward"],
                "precision": uns_summary["precision"],
                "recall": uns_summary["recall"],
                "f1": uns_summary["f1"],
            }
        )
    else:
        print(f"[WARN] Unsloth TRL model not found at: {unsloth_path}")

    _save_rows_csv(all_rows, args.output_csv)

    print("\n" + "=" * 52)
    print("TRLEX BENCHMARK REPORT")
    print("=" * 52)
    print(f"{'Policy':22s} {'Avg Reward':>11s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
    for row in report_rows:
        print(
            f"{row['policy'][:22]:22s} "
            f"{row['avg_reward']:11.2f} "
            f"{row['precision']:10.2f} "
            f"{row['recall']:8.2f} "
            f"{row['f1']:8.2f}"
        )
    print("=" * 52)
    print(f"Per-episode metrics saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TradeX TRL policies and export metrics.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model_path", type=str, default=DEFAULT_TRL_PATH)
    parser.add_argument("--unsloth_model_path", type=str, default=DEFAULT_UNSLOTH_PATH)
    parser.add_argument("--output_csv", type=str, default="outputs/trl_eval_metrics.csv")
    main(parser.parse_args())
