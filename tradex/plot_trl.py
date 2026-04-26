import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_by_policy(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["policy"]].append(r)
    return grouped


def _float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _latest_training_history(outputs_dir: str) -> List[Dict]:
    candidates = glob.glob(os.path.join(outputs_dir, "training_history_*.json"))
    if not candidates:
        candidates = glob.glob(os.path.join("models", "trl_overseer", "training_history_*.json"))
    if not candidates:
        return []

    latest = sorted(candidates)[-1]
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_reward_vs_episode(grouped: Dict[str, List[Dict]], out_path: str):
    plt.figure(figsize=(12, 6))
    for policy, rows in grouped.items():
        eps = [_float(r["episode"]) for r in rows]
        rews = [_float(r["reward"]) for r in rows]
        if not eps:
            continue
        order = np.argsort(eps)
        eps = np.array(eps)[order]
        rews = np.array(rews)[order]
        plt.plot(eps, rews, alpha=0.25, linewidth=1)
        if len(rews) >= 10:
            smooth = np.convolve(rews, np.ones(10) / 10, mode="valid")
            plt.plot(eps[9:], smooth, linewidth=2, label=policy)
        else:
            plt.plot(eps, rews, linewidth=2, label=policy)
    plt.title("TRL Reward vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_precision_recall(grouped: Dict[str, List[Dict]], out_path: str):
    plt.figure(figsize=(12, 6))
    policies = list(grouped.keys())
    p = [np.mean([_float(r["precision"]) for r in grouped[k]]) for k in policies]
    r = [np.mean([_float(r["recall"]) for r in grouped[k]]) for k in policies]

    x = np.arange(len(policies))
    w = 0.35
    plt.bar(x - w / 2, p, width=w, label="Precision")
    plt.bar(x + w / 2, r, width=w, label="Recall")
    plt.xticks(x, policies, rotation=15, ha="right")
    plt.ylabel("Percentage")
    plt.title("TRL Precision vs Recall by Policy")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_action_distribution(grouped: Dict[str, List[Dict]], out_path: str):
    plt.figure(figsize=(12, 6))
    policies = list(grouped.keys())
    allow = [np.mean([_float(r["allow_rate"]) for r in grouped[k]]) for k in policies]
    block = [np.mean([_float(r["block_rate"]) for r in grouped[k]]) for k in policies]

    x = np.arange(len(policies))
    plt.bar(x, allow, label="Allow Rate")
    plt.bar(x, block, bottom=allow, label="Block Rate")
    plt.xticks(x, policies, rotation=15, ha="right")
    plt.ylabel("Percentage")
    plt.title("TRL Action Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_loss_curve(history: List[Dict], out_path: str):
    plt.figure(figsize=(12, 6))
    if history:
        eps = [h.get("episode", i) for i, h in enumerate(history)]
        rewards = np.array([_float(h.get("reward", 0.0)) for h in history], dtype=np.float32)
        # PPO trainer history in this repo does not persist explicit loss.
        # We derive a stable proxy from normalized negative reward trend.
        if rewards.std() > 1e-6:
            proxy_loss = -(rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            proxy_loss = -rewards
        plt.plot(eps, proxy_loss, label="Loss Proxy (reward-derived)", linewidth=2)
    else:
        plt.plot([0, 1], [0, 0], label="No training history found", linewidth=2)
    plt.title("TRL Loss Curve")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_ppo_vs_trl(grouped: Dict[str, List[Dict]], out_path: str):
    plt.figure(figsize=(12, 6))
    policies = list(grouped.keys())
    avg_reward = [np.mean([_float(r["reward"]) for r in grouped[k]]) for k in policies]
    f1 = [np.mean([_float(r["f1"]) for r in grouped[k]]) for k in policies]

    x = np.arange(len(policies))
    w = 0.35
    plt.bar(x - w / 2, avg_reward, width=w, label="Avg Reward")
    plt.bar(x + w / 2, f1, width=w, label="F1")
    plt.xticks(x, policies, rotation=15, ha="right")
    plt.title("PPO vs TRL Policy Comparison")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main(args):
    os.makedirs(args.plots_dir, exist_ok=True)
    rows = read_csv_rows(args.eval_csv)
    if not rows:
        raise FileNotFoundError(
            f"No evaluation rows found at {args.eval_csv}. Run `python -m tradex.eval_trl` first."
        )

    grouped = _group_by_policy(rows)
    history = _latest_training_history(args.outputs_dir)

    plot_reward_vs_episode(grouped, os.path.join(args.plots_dir, "trl_reward_vs_episode.png"))
    plot_precision_recall(grouped, os.path.join(args.plots_dir, "trl_precision_recall.png"))
    plot_action_distribution(grouped, os.path.join(args.plots_dir, "trl_action_distribution.png"))
    plot_loss_curve(history, os.path.join(args.plots_dir, "trl_loss_curve.png"))
    plot_ppo_vs_trl(grouped, os.path.join(args.plots_dir, "ppo_vs_trl_bar.png"))

    print(f"Saved TRL plots to: {args.plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate polished TRL evaluation plots.")
    parser.add_argument("--eval_csv", type=str, default="outputs/trl_eval_metrics.csv")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--plots_dir", type=str, default="plots")
    main(parser.parse_args())
