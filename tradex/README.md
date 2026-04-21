# TradeX — Multi-Agent Market Simulation with Scalable Oversight

## What it does

TradeX is a sophisticated multi-agent reinforcement learning environment simulating a financial market where four autonomous trading agents interact within a shared economy. A special Overseer Agent, powered by LangGraph, monitors agent behavior through observable signals only, inferring malicious intent without accessing hidden strategies. The system trains an RL policy to intervene by blocking suspicious agents, promoting market stability and fairness. This project demonstrates emergent behaviors from agent interactions, theory-of-mind inference, and scalable oversight in multi-agent systems.

## Architecture

The simulation consists of:
- **4 Trading Agents**: Each with unique, hidden strategies and internal states
  - NormalTrader (agent 0): Mean-reversion trader
  - ManipulatorBot (agent 1): Pump-and-dump schemer (malicious)
  - ArbitrageAgent (agent 2): Stabilizing arbitrageur
  - LiquidityProvider (agent 3): Passive market maker
- **Overseer Agent**: LangGraph-based system with observe → analyze → decide → intervene nodes
- **Market Environment**: Gym-compatible with price dynamics, trade execution, and reward computation
- **RL Training Pipeline**: Policy gradient training to learn optimal intervention strategies

Agents interact through price impacts, creating emergent market dynamics that the Overseer must navigate.

## How to run

```bash
pip install -r requirements.txt
python tradex/train.py        # Trains the overseer policy (200 episodes)
python tradex/compare.py      # Runs comparison between with/without overseer
```

Training takes ~5-10 minutes on a standard CPU. The trained policy is saved as `policy.pth`.

## Files

| File | Purpose |
|------|---------|
| `env.py` | MarketEnv class — Gym-compatible environment |
| `agents.py` | Four trading agent classes with hidden strategies |
| `overseer.py` | LangGraph Overseer Agent implementation |
| `graph.py` | LangGraph orchestration of one simulation timestep |
| `reward.py` | Reward function for RL training |
| `train.py` | Policy gradient training loop |
| `compare.py` | Before/after comparison script |
| `utils.py` | Logging and plotting utilities |
| `README.md` | This documentation |

## Theme alignment

This project directly addresses the **Multi-Agent Interactions + Scalable Oversight (Fleet AI)** theme:

- **Multi-Agent Interactions**: Agents' trades create complex price dynamics and emergent behaviors (e.g., ManipulatorBot's pump triggering ArbitrageAgent's stabilization attempts)
- **Scalable Oversight**: The Overseer uses LangGraph to process multi-modal signals (price trends, trade volumes, correlations) and makes theory-of-mind inferences about agent intent, scaling from 4 to potentially hundreds of agents
- **Fleet AI**: Demonstrates hierarchical oversight where a central AI monitors and intervenes in a fleet of autonomous agents, ensuring system-level objectives (market stability) over individual agent goals