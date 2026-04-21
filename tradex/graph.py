from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from .agents import NormalTrader, ManipulatorBot, ArbitrageAgent, LiquidityProvider
from .overseer import Overseer
from .reward import compute_reward
import numpy as np

class SimulationState(TypedDict):
    price: float
    price_history: List[float]
    trade_history: List[dict]
    timestep: int
    blocked_agents: List[int]
    trades: List[dict]
    overseer_action: str
    reward: float
    done: bool

# Agent instances
agents = [NormalTrader(0), ManipulatorBot(1), ArbitrageAgent(2), LiquidityProvider(3)]
overseer = Overseer()

def agent_0_node(state: SimulationState) -> SimulationState:
    if 0 not in state["blocked_agents"]:
        trade = agents[0].act(state["price"])
        state["trades"].append(trade)
    return state

def agent_1_node(state: SimulationState) -> SimulationState:
    if 1 not in state["blocked_agents"]:
        trade = agents[1].act(state["price"])
        state["trades"].append(trade)
    return state

def agent_2_node(state: SimulationState) -> SimulationState:
    if 2 not in state["blocked_agents"]:
        trade = agents[2].act(state["price"])
        state["trades"].append(trade)
    return state

def agent_3_node(state: SimulationState) -> SimulationState:
    if 3 not in state["blocked_agents"]:
        trade = agents[3].act(state["price"])
        state["trades"].append(trade)
    return state

def overseer_node(state: SimulationState) -> SimulationState:
    observation = {
        "price": state["price"],
        "price_history": state["price_history"],
        "timestep": state["timestep"],
        "trade_history": state["trade_history"],
        "blocked_agents": state["blocked_agents"],
        "price_deviation": state["price"] - 100.0
    }
    action, reasoning = overseer.act(observation)
    state["overseer_action"] = action
    state["reasoning"] = reasoning
    return state

def env_step_node(state: SimulationState) -> SimulationState:
    trades = state["trades"]
    # Update price
    buy_volume = sum(t["size"] for t in trades if t["action"] == "buy")
    sell_volume = sum(t["size"] for t in trades if t["action"] == "sell")
    price_change = (buy_volume - sell_volume) * 0.1
    noise = np.random.normal(0, 0.2)
    new_price = state["price"] + price_change + noise
    new_price = np.clip(new_price, 50, 200)
    state["price"] = new_price

    # Apply overseer action
    if state["overseer_action"].startswith("BLOCK_"):
        agent_id = int(state["overseer_action"].split("_")[1])
        state["blocked_agents"].append(agent_id)

    # Compute reward
    reward = compute_reward(state["overseer_action"], trades, new_price, set(state["blocked_agents"]), 100.0)
    state["reward"] = reward

    # Update histories
    state["price_history"].append(new_price)
    if len(state["price_history"]) > 10:
        state["price_history"].pop(0)
    state["trade_history"].extend(trades)
    if len(state["trade_history"]) > 5:
        state["trade_history"] = state["trade_history"][-5:]

    state["timestep"] += 1
    state["done"] = state["timestep"] >= 50  # Assuming num_steps=50

    return state

# Build graph
graph = StateGraph(SimulationState)
graph.add_node("agent_0", agent_0_node)
graph.add_node("agent_1", agent_1_node)
graph.add_node("agent_2", agent_2_node)
graph.add_node("agent_3", agent_3_node)
graph.add_node("overseer", overseer_node)
graph.add_node("env_step", env_step_node)

# Parallel agents
graph.add_edge(START, "agent_0")
graph.add_edge(START, "agent_1")
graph.add_edge(START, "agent_2")
graph.add_edge(START, "agent_3")

# After agents, to overseer
graph.add_edge("agent_0", "overseer")
graph.add_edge("agent_1", "overseer")
graph.add_edge("agent_2", "overseer")
graph.add_edge("agent_3", "overseer")

# Then env_step
graph.add_edge("overseer", "env_step")
graph.add_edge("env_step", END)

compiled_graph = graph.compile()