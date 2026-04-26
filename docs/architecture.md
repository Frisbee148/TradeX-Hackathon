# TradeX / MEVerse — System Architecture

End-to-end map of the codebase: **two coupled stacks** sharing one repo.

- **`tradex/`** → multi-agent AMM simulator + PPO Overseer (research/training stack).
- **`meverse/`** → OpenEnv-compliant single-agent surveillance benchmark (evaluation/deployment stack).
- **`app.py` / `dashboard.py`** → Gradio UIs that wrap both stacks.
- **`inference.py`** → LLM (HF router) policy runner against `meverse`.
- **`server/`, `meverse/server/`** → FastAPI hosts (OpenEnv runtime + HF Space mode).

---

## 1. Top-level layout

```
final/
├── tradex/              # Multi-agent AMM + PPO Overseer (training)
├── meverse/             # OpenEnv benchmark (evaluation)
│   └── server/          # FastAPI app, environment class
├── server/              # Thin OpenEnv server entry
├── app.py               # TradeX Gradio dashboard (uses tradex/)
├── dashboard.py         # MEVerse Gradio dashboard (uses meverse/)
├── inference.py         # LLM-driven policy runner against meverse
├── client.py            # Re-exports MeverseAction/Env/Observation
├── compare_policies.py  # CLI benchmark wrapper
├── trl_demo.py          # TRL prototype (planned LLM finetune path)
├── openenv.yaml         # Benchmark spec (3 tasks, FastAPI runtime)
├── Dockerfile           # uv-based image, openenv-base
├── models/best_model.pth, policy.pth   # Trained PPO checkpoints
├── telemetry/           # JSONL run logs (replayed by dashboard tab 3)
└── docs/                # Dashboard.md, playground.md, hf-mini-blog.md, this file
```

---

## 2. Two stacks side-by-side

| Aspect | `tradex/` (PPO research) | `meverse/` (OpenEnv benchmark) |
|---|---|---|
| Env class | `tradex.env.MarketEnv` | `meverse.server.meverse_environment.MarketSurveillanceEnvironment` |
| Action space | `ALLOW`, `BLOCK_<agent_id>` (5 discrete) | `ALLOW`, `MONITOR`, `FLAG`, `BLOCK` (4 discrete) |
| Observation | dict (54-dim vector after `encode_observation`) | `SurveillanceObservation` Pydantic (16+ fields) |
| Agents | 4 strategic agents per episode | Procedural bot vs noisy-normal generator |
| AMM | constant-product `x·y=k`, `reserve_y=100000`, baseline price 100 | constant-product `x·y=k`, `reserve_x=1000`, `reserve_y=100000` |
| Reward | TP `+3`, wrong target `-1.5`, FN `-3`, FP `-0.7`, TN `+0.05` | severity-shaped `[0..1]`, see `_reward_for_action` |
| Policy | `tradex.overseer.Overseer` (PPO, hierarchical actor + critic) | `meverse.baseline_policy.choose_surveillance_action` (threshold rules) + LLM via `inference.py` |
| Training | `tradex/train.py` (PPO + GAE + supervised pretrain + collapse-recovery) | none in repo (env + grader only) |
| Grader | reward signal | `meverse.tasks.compute_task_grade` (5-component weighted score) |
| Tasks | curriculum stages 1–5 | `burst_detection` / `pattern_manipulation_detection` / `full_market_surveillance` |

---

## 3. `tradex/` — multi-agent training stack

### Agents (`tradex/agents.py`)

| Class | Role | Behavior |
|---|---|---|
| `NormalTrader` | Mean-reversion / momentum | FOMO-buy on +momentum, panic-sell on −momentum |
| `Arbitrage` | Stabilizer | Buys when price < 96, sells when > 104 |
| `Manipulator` | Adversary (`is_malicious=True`) | 4 strategies (A/B/C/D): pump-dump, stealth accumulation, burst momentum, flash crash. Strategy mix scales with `stage` (curriculum). |
| `NoisyTrader` | Benign decoy | Random ±large trades that mimic attacks |

`MarketEnv` always seats 1 Manipulator + 3 random benign agents.

### Environment (`tradex/env.py`)

```
reset(stage, seed)
    └─► sample agents, init AMM (price=100), warm price_history,
        generate_next_intents() → threat_score via _calculate_threat()

step(overseer_action)
    ├─► parse BLOCK_<id>, drop blocked agent's intent
    ├─► execute remaining trades against AMM (constant product)
    ├─► update agent_stats (burst, pump, dump, recent_actions, pnl)
    ├─► reward = compute_reward(...)  (tradex/reward.py)
    └─► generate_next_intents() for next observation (preview)
```

`_calculate_threat` is a hand-coded rule engine (4 rules) that produces the `threat_score` fed into the policy and used as a reward gate.

### Policy network (`tradex/overseer.py`)

```
obs_vec (54-dim)
    │
    ▼
[Linear → LayerNorm → ReLU] × 3   (256 → 256 → 128)
    │
    ├──► actor_intervene  (2 logits: ALLOW vs BLOCK)
    │       └─ +threat_bias: high threat suppresses ALLOW (×−15),
    │          low threat mildly suppresses BLOCK (×−3)
    ├──► actor_target     (4 logits: which agent to block)
    └──► critic           (V(s))

probs = [P(ALLOW), P(BLOCK_0), P(BLOCK_1), P(BLOCK_2), P(BLOCK_3)]
        = [P(allow), P(block) · P(target_i)]   ← hierarchical mixture
```

`encode_observation` packs: price, momentum, volatility, log liquidity, price-dev, timestep, stage, threat_score, 10 price-history values, then 9 stat features × 4 agents.

### Training (`tradex/train.py`)

1. **Hard gate** — `validate_detector(env)` runs 150 rule-only episodes. If precision < 60% or recall < 70%, abort. PPO will not start on a broken detector.
2. **Supervised pretrain** — 250 episodes × 50 steps of synthetic states, label = `BLOCK_manip` if `threat>0.85` else `ALLOW`. CE loss for 8 epochs.
3. **PPO loop** — `compute_gae(γ=0.99, τ=0.95)`, clip 0.2, 4 epochs × mini-batch 64, entropy schedule `max(0.02, 0.05·(1−ep/total))`, grad-norm clip 0.5, StepLR.
4. **Curriculum** — `get_stage(episode)` ramps stage 1→5 over 1800+ episodes.
5. **Collapse recovery** — 50 consecutive episodes with `ALLOW%>95` triggers checkpoint reload + fresh optimizer.
6. **Best-model criterion** — `score = avg_reward + 4·F1 + 2·recall` (with non-zero precision and recall).

Outputs: `models/best_model.pth`, `metrics/training_history.json`, `plots/*.png`, per-episode `logs/episode_<n>_seed_<s>.json`.

### Reward (`tradex/reward.py`)

| Truth | Action | Reward | Tag |
|---|---|---|---|
| attack active | `BLOCK_<malicious>` | +3.0 | TP |
| attack active | `BLOCK_<benign>` | −1.5 | wrong target |
| attack active | `ALLOW` | −3.0 | FN |
| no attack | `BLOCK_*` | −0.7 | FP |
| no attack | `ALLOW` | +0.05 | TN |

Attack flag fires only when summed manipulator size ≥ 15.0.

### Eval (`tradex/compare.py`, `compare_generalization.py`)

`run_evaluation` runs N=50 unseen-seed episodes and reports avg reward, price error, precision/recall/F1, intervention rate, action distribution. Compares Heuristic / PPO-Det / PPO-Stoch / Rule-Hybrid.

---

## 4. `meverse/` — OpenEnv benchmark stack

### AMM core (`meverse/amm.py`)

`AMMState(reserve_x=1000, reserve_y=100000, bot_confidence, volatility, health_index, step)`.

`apply_action_effects` is a state machine: action × label → state delta (e.g., `BLOCK` on suspicious shrinks `bot_confidence`/`volatility`; `BLOCK` on normal hurts `health_index`/`volatility`). This creates the **feedback loop**: today's action shapes tomorrow's distribution.

`generate_step_from_state(state, rng, profile)` is a procedural generator. With probability `bot_confidence`, the step is a bot step (overt or stealthy depending on `state.step`). Otherwise it's an organic step with a 15% noise-spike chance. Profile (burst / pattern / mixed) biases the indicator distribution.

### Tasks (`meverse/tasks.py`)

```
TASK_DEFINITIONS = {
  burst_detection                : 50 steps, init bot_conf=0.25, profile=burst
  pattern_manipulation_detection : 50 steps, init bot_conf=0.35, profile=pattern
  full_market_surveillance       : 60 steps, init bot_conf=0.30, profile=mixed
}
```

`compute_task_grade(task, actions, labels)` returns a 5-component weighted score:

```
score = 0.50 detection
      + 0.20 false_positive
      + 0.15 false_negative
      + 0.10 health
      + 0.05 overblocking          (clamped strictly to (0, 1))
```

Per-step credit on suspicious labels: `BLOCK=1.0`, `FLAG=0.82`, `MONITOR=0.55`, `ALLOW=0.05`.

### Environment (`meverse/server/meverse_environment.py`)

Subclass of `openenv.core.env_server.interfaces.Environment`. Implements:

- `reset(seed, episode_id, task=...)` — fresh `AMMState` + `Random(seed)`, generate first step.
- `step(action: SurveillanceAction)` — validate, score reward, push label/action/reward, `apply_action_effects`, increment, generate next step (unless done).
- `grade()` — calls `compute_task_grade` plus baseline-policy hint.
- `debug_snapshot()` — for telemetry (used by `inference.py` JSONL).
- `eval_mode` / `demo_mode` — fixed seed=42 vs random; controlled by env vars.

### Models (`meverse/models.py`)

`SurveillanceAction(action_type ∈ {ALLOW, FLAG, BLOCK, MONITOR})` with case-normalizing validator.
`SurveillanceObservation` — 16+ float/int fields covering window stats, indicators, step counter, task name, plus `metadata` dict.

### Baseline policy (`meverse/baseline_policy.py`)

Pure threshold rules (no learning):

```
pattern≥0.72 ∧ slippage≥0.055   → BLOCK
manipulation≥0.78               → BLOCK
burst≥0.70 ∨ trade_freq≥7.5     → FLAG
suspiciousness≥0.52             → MONITOR
otherwise                       → ALLOW
```

This is the lower-bound benchmark and the LLM fallback in `inference.py`.

---

## 5. App / UI layer

### `dashboard.py` — MEVerse dashboard (Gradio)

4 tabs: Episode Runner, Policy Comparison, Telemetry Viewer, About. Wraps `MarketSurveillanceEnvironment` with `Heuristic / Always-Allow / Random` policy selectors. Renders 9+ Plotly visualizations (gauges, reward timeline, signal heatmap, AMM evolution, grade radar, confusion matrix, step table). Theme is custom dark-teal, no purple. Full per-chart spec in `docs/Dashboard.md`.

### `app.py` — TradeX dashboard (Gradio)

3 tabs: Live Market Replay (PPO Overseer episode), Reward Optimization Curves (loads `plots/*.png`), Baseline vs Learned Policy (calls `tradex.compare.run_evaluation` for 4 modes).

### HF Space mode

`SPACE_ID` env var detected → `_build_space_app()` (in repo's `app.py` orchestration) merges OpenEnv Playground (auto-generated from `SurveillanceAction` Pydantic schema) and the dashboard into one `gr.Tabs` Gradio app. Falls back to dashboard-only if the OpenEnv web-interface import fails.

### `inference.py` — LLM policy runner

Drives `MarketSurveillanceEnvironment` step-by-step using an OpenAI-compatible client pointed at `https://router.huggingface.co/v1` (model: `Qwen/Qwen2.5-72B-Instruct` by default). System prompt embeds the same threshold rules as `baseline_policy`. On parse failure or missing `HF_TOKEN`, it falls back to `choose_surveillance_action`. Optional `DEBUG_TELEMETRY=1` writes `telemetry/<task>-<utc>.jsonl` (replayable in dashboard tab 3).

---

## 6. Serving layer

### OpenEnv FastAPI (`meverse/server/app.py`)

```python
app = create_app(
    MarketSurveillanceEnvironment,
    SurveillanceAction,
    SurveillanceObservation,
    env_name="amm-market-surveillance",
    max_concurrent_envs=1,
)
```

Routes:
- `POST /reset`, `POST /step`, `POST /grade` (OpenEnv contract)
- `/docs` (FastAPI Swagger; redirected from `/` when not in HF Space)
- `/` mounted as Gradio when `SPACE_ID` is present

### `server/app.py` — alias entry for `openenv validate`

### `openenv.yaml`

`spec_version: 1, runtime: fastapi, app: app:app, port: 7860`. Declares the 3 tasks with `grader: programmatic`.

### `Dockerfile`

Two-stage build on `ghcr.io/meta-pytorch/openenv-base`. Uses `uv sync --frozen --no-editable` (twice — once without, once with the project) inside `meverse/`, then installs root `requirements.txt` (gradio, plotly, numpy) into the same venv. Final image runs `python app.py`. `PYTHONPATH=/app/env:/app/env/meverse`.

---

## 7. Data flow — end-to-end

```
                 ┌─────────────────────────────────────────────────┐
   training      │  tradex.env.MarketEnv  ←→  Overseer (PyTorch)   │
   path          │  4 strategic agents       PPO + GAE + pretrain  │
                 │  rule-engine threat       best_model.pth        │
                 └─────────────────────────────────────────────────┘
                                       │
                                       ▼  (insights inform thresholds, not weights)
                 ┌─────────────────────────────────────────────────┐
   benchmark     │  meverse.MarketSurveillanceEnvironment          │
   path          │  AMMState + procedural step generator           │
                 │  apply_action_effects feedback loop             │
                 │  compute_task_grade  (5-component weighted)     │
                 └─────────────────────────────────────────────────┘
                            ▲                          ▲
                            │                          │
       ┌────────────────────┴───────────┐    ┌─────────┴──────────┐
       │ inference.py (LLM via HF       │    │ dashboard.py       │
       │  router; baseline fallback)    │    │  Heuristic/Random/ │
       │  → telemetry/*.jsonl           │    │  Always-Allow      │
       └────────────────────┬───────────┘    └─────────┬──────────┘
                            │                          │
                            ▼                          ▼
                 ┌─────────────────────────────────────────────────┐
                 │  FastAPI (openenv-core)  +  Gradio (HF Space)   │
                 │  /reset /step /grade  +  Playground / Dashboard │
                 └─────────────────────────────────────────────────┘
```

---

## 8. Key extension points

| Need | Touch |
|---|---|
| New surveillance task | `meverse/amm.py: TASK_CONFIGS` + `meverse/tasks.py: TASK_DEFINITIONS` |
| New adversary strategy | `tradex/agents.py: Manipulator.act` (curriculum-aware switch) |
| New rule in threat detector | `tradex/env.py: MarketEnv._calculate_threat` |
| New reward shaping | `tradex/reward.py` (PPO) or `meverse_environment._reward_for_action` (benchmark) |
| New policy under test | `meverse/baseline_policy.py` (heuristic) or `inference.py: select_action` (LLM) |
| New dashboard chart | `dashboard.py: _make_*_chart` + wire into `run_full_episode` outputs |
| TRL / Unsloth path | `trl_demo.py` (prototype), `tradex/train_trl.py`, `tradex/train_trl_unsloth.py` |

---

## 9. Runtime modes

| Mode | Trigger | Stack |
|---|---|---|
| Local OpenEnv | `python app.py` (no SPACE_ID) | FastAPI on :7860, `/docs` |
| HF Space | `SPACE_ID` set | Gradio Playground + Dashboard tabs at `/` |
| Standalone dash | `python dashboard.py` | Gradio Blocks only |
| Training | `python -m tradex.train --episodes N` | PPO loop, no server |
| Eval | `python -m tradex.compare_generalization` | CLI benchmark |
| LLM benchmark | `python inference.py` (with `HF_TOKEN`) | OpenAI client → meverse env |
