# TradeX + MEVerse

TradeX started from a simple question: can we train an agent to intervene in AMM markets before manipulation cascades into full instability?

This repo now contains two connected systems:

- `tradex/`: a multi-agent AMM simulator with a PPO-trained overseer (research + training path).
- `meverse/`: an OpenEnv-compliant market-surveillance benchmark (evaluation + deployment path).

If you only read one extra page after this README, read the companion HF blog: [`docs/hf-mini-blog.md`](docs/hf-mini-blog.md).

## 1) Problem: the capability gap

Most "anomaly detection" settings are static classification tasks. Real markets are not static.

In AMM environments, one action changes price, liquidity, and incentives for every other actor. That means surveillance is not just detection. It is sequential governance under uncertainty:

- adversarial behavior can look normal for several steps,
- false positives can damage healthy flow,
- delayed intervention can amplify volatility.

TradeX/MEVerse targets that gap: train and benchmark agents that make intervention decisions over time, not one-shot labels.

## 2) Environment: what the agent sees, does, and gets rewarded for

### What the agent sees

In `meverse/`, each step provides a structured observation with market micro-signals:

- AMM price and liquidity snapshots,
- burst and pattern indicators,
- trade frequency and slippage proxies,
- suspiciousness/manipulation scores,
- episode progress metadata.

### What the agent does

In the OpenEnv benchmark (`meverse/server/meverse_environment.py`), the action space is:

- `ALLOW`
- `MONITOR`
- `FLAG`
- `BLOCK`

Each action feeds back into future state through AMM dynamics (`meverse/amm.py`), so behavior today shifts tomorrow's distribution.

### How reward and grading work

- Step reward is shaped by action correctness and severity (`_reward_for_action`).
- Episode-level grading comes from `compute_task_grade` in `meverse/tasks.py`.
- Tasks declared in `openenv.yaml`:
  - `burst_detection`
  - `pattern_manipulation_detection`
  - `full_market_surveillance`

## 3) Results: what changed after training

The current "trained" component in this repo is the `tradex/` PPO overseer.

- PPO training pipeline: `tradex/train.py`
- Generalization benchmark: `python -m tradex.compare_generalization`
- Comparison utilities: `tradex/compare.py`, `compare_policies.py`
- Best checkpoint location: `models/best_model.pth`

In practice, training changes behavior from mostly passive allowing to targeted intervention under high-threat trajectories, with measurable movement in precision/recall/F1 and reward profiles in benchmark runs.

The OpenEnv side (`meverse/`) is the benchmark-ready environment and grader used for agent evaluation and HF deployment.

## 4) Why it matters

Who cares:

- researchers building RL/LLM agents for dynamic risk control,
- teams shipping guardrail policies for high-frequency systems,
- evaluators who need environments where actions alter future risk.

Why:

- It is a compact testbed for "monitoring as control" rather than static labeling.
- It supports both near-term baselines and longer-term LLM training pathways.
- It is reproducible enough for comparison, but rich enough to exhibit strategic/adaptive behavior.

## Architecture at a glance

```text
tradex/  -> multi-agent AMM + PPO overseer training/eval
meverse/ -> OpenEnv environment + grading + FastAPI serving
app.py   -> Gradio UI for TradeX stack
dashboard.py -> Gradio UI for MEVerse stack
inference.py -> LLM policy runner against MEVerse
openenv.yaml -> OpenEnv manifest and task registry
```

For full technical mapping, see [`docs/architecture.md`](docs/architecture.md).

## Engineering table-stakes checklist

This repo currently satisfies the requested implementation constraints:

- Uses OpenEnv `Environment` base class in `meverse/server/meverse_environment.py`.
- Maintains client/server separation (`meverse/client.py` does not import server internals).
- Implements Gym-style core lifecycle (`reset`, `step`, `state`).
- Includes valid `openenv.yaml` manifest with task definitions.
- MCP/OpenEnv tool names avoid reserved runtime endpoints (`reset`, `step`, `state`, `close`).

## Run locally

```bash
pip install -r requirements.txt

# Train PPO overseer (TradeX stack)
python -m tradex.train --episodes 1000

# Evaluate generalization on unseen seeds
python -m tradex.compare_generalization

# Run OpenEnv/FastAPI app (serves app:app as declared in openenv.yaml)
python server/app.py

# Optional UIs
python app.py
python dashboard.py
```

## Links

- HF mini blog: [`docs/hf-mini-blog.md`](docs/hf-mini-blog.md)
- Architecture deep dive: [`docs/architecture.md`](docs/architecture.md)

If you publish a demo video or HF post, add it here so reviewers can jump directly from the README.
