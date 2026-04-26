# From Alarm Bells to Agent Policy: Building a Market Surveillance Environment

Companion to the project README: [`../README.md`](../README.md)

## The story in one paragraph

AMM markets do not fail because one bad trade appears; they fail because suspicious behavior compounds while everyone reacts. We built TradeX + MEVerse to model that compounding effect and train agents to make intervention decisions under uncertainty. Instead of asking "is this trade malicious?", we ask "what action now keeps the next 20 steps healthier?"

## 1) Problem

We target a practical capability gap: sequential market surveillance.

Most benchmarks stop at static detection labels. Real governance needs policies that balance:

- catching manipulation early,
- avoiding unnecessary blocks,
- preserving market health across time.

The domain here is adversarial AMM trading with burst and pattern manipulations that can resemble normal activity until context accumulates.

## 2) Environment

The benchmarkable environment is `meverse/`, served through OpenEnv/FastAPI.

- **Agent observations:** AMM price/liquidity, trade burst/pattern indicators, slippage and frequency features, suspiciousness/manipulation scores.
- **Action space:** `ALLOW`, `MONITOR`, `FLAG`, `BLOCK`.
- **Reward loop:** action quality is rewarded per-step, and actions update AMM state so future observations change.
- **Tasks:** `burst_detection`, `pattern_manipulation_detection`, `full_market_surveillance` (declared in `openenv.yaml`).

Under the hood, this uses the OpenEnv `Environment` base class in `meverse/server/meverse_environment.py` and standard lifecycle methods (`reset`, `step`, `state`).

## 3) Results

What is trained today is the `tradex/` PPO overseer stack; what is deployment-ready today is the `meverse/` OpenEnv benchmark stack.

- PPO training pipeline exists and runs from `tradex/train.py`.
- Generalization benchmarking is available via `python -m tradex.compare_generalization`.
- Trained checkpoints are stored in `models/`.
- In observed runs, policy behavior shifts from permissive defaults toward targeted interventions on high-threat episodes, improving task-relevant metrics over baselines.

This split is intentional: `tradex/` explores training dynamics, while `meverse/` provides a clean benchmark contract for evaluation and HF deployment.

## 4) Why it matters

This matters to anyone evaluating whether LLM/RL agents can do reliable governance in dynamic systems:

- **Researchers:** a compact environment where action changes future risk.
- **Builders:** a practical testbed for surveillance policies before production.
- **Reviewers/Judges:** a setup that is ambitious but grounded in runnable artifacts and clear interfaces.

If we want frontier-capable agents, we need environments where they are judged on long-horizon decisions, not one-shot labels.

## Engineering quality notes

Current implementation checks the table-stakes boxes:

- OpenEnv base classes are used correctly.
- Client/server separation is respected (`meverse/client.py` vs `meverse/server/`).
- Gym-style lifecycle is implemented.
- `openenv.yaml` is valid and task-complete.
- Reserved endpoint/tool names are not reused for custom MCP tools.

## Try it

```bash
pip install -r requirements.txt
python -m tradex.compare_generalization
python server/app.py
python dashboard.py
```

## Links to include when publishing

- Project README: [`../README.md`](../README.md)
- Architecture deep dive: [`architecture.md`](architecture.md)
- Add your Hugging Face Space URL here
- Add your demo video URL here
