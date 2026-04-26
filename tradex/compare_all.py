import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .compare import run_evaluation
from .eval_trl import DEFAULT_TRL_PATH, DEFAULT_UNSLOTH_PATH, evaluate_model_path


def _to_market_stability(avg_price_error: float) -> float:
    # Higher is better.
    return max(0.0, 100.0 - (float(avg_price_error) * 10.0))


def _row_from_eval(name: str, metrics: Dict) -> Dict:
    return {
        "policy": name,
        "avg_reward": float(metrics.get("avg_reward", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1_score", metrics.get("f1", 0.0))),
        "false_positives": float(metrics.get("false_positives", 0.0)),
        "missed_attacks": float(metrics.get("missed_attacks", metrics.get("false_negatives", 0.0))),
        "market_stability": _to_market_stability(float(metrics.get("avg_final_price_error", metrics.get("avg_price_error", 0.0)))),
        "intervention_rate": float(metrics.get("intervention_rate", 0.0)),
    }


def compare_all(episodes: int, trl_model_path: str, unsloth_model_path: str) -> Tuple[pd.DataFrame, List[Dict]]:
    rows: List[Dict] = []

    heuristic = run_evaluation(num_episodes=episodes, use_overseer=True, pure_rule_based=True)
    rows.append(_row_from_eval("Heuristic baseline", heuristic))

    ppo_det = run_evaluation(num_episodes=episodes, use_overseer=True, deterministic=True)
    rows.append(_row_from_eval("PPO deterministic", ppo_det))

    ppo_stoch = run_evaluation(num_episodes=episodes, use_overseer=True, deterministic=False)
    rows.append(_row_from_eval("PPO stochastic", ppo_stoch))

    if os.path.exists(trl_model_path):
        _, trl_summary = evaluate_model_path(trl_model_path, "TRL model", episodes)
        rows.append(_row_from_eval("TRL model", trl_summary))
    else:
        rows.append(_row_from_eval("TRL model (missing)", {}))

    if os.path.exists(unsloth_model_path):
        _, uns_summary = evaluate_model_path(unsloth_model_path, "TRL Unsloth model", episodes)
        rows.append(_row_from_eval("TRL Unsloth model", uns_summary))
    else:
        rows.append(_row_from_eval("TRL Unsloth model (missing)", {}))

    df = pd.DataFrame(rows)
    return df, rows


def _save_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    df, rows = compare_all(args.episodes, args.model_path, args.unsloth_model_path)
    _save_csv(args.output_csv, rows)

    print("\nFINAL BENCHMARK")
    print(df.to_string(index=False))
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PPO and TRL policy variants.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model_path", type=str, default=DEFAULT_TRL_PATH)
    parser.add_argument("--unsloth_model_path", type=str, default=DEFAULT_UNSLOTH_PATH)
    parser.add_argument("--output_csv", type=str, default="outputs/final_benchmark.csv")
    main(parser.parse_args())
