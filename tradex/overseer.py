from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END

class OverseerState(TypedDict):
    price: float
    price_history: List[float]
    trade_history: List[dict]
    timestep: int
    blocked_agents: List[int]
    suspicion_scores: Dict[int, float]
    decision: str
    reasoning: str

def observe_node(state: OverseerState) -> OverseerState:
    # Compute per-agent trade volume in last 5 steps
    trade_volumes = {i: 0.0 for i in range(4)}
    for trade in state["trade_history"][-5:]:
        agent_id = trade["agent_id"]
        trade_volumes[agent_id] += trade["size"]

    # Compute price trend
    if len(state["price_history"]) >= 2:
        recent_prices = state["price_history"][-5:]
        if recent_prices[-1] > recent_prices[0] + 2:
            trend = "rising"
        elif recent_prices[-1] < recent_prices[0] - 2:
            trend = "falling"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # Update state with signals (though not directly used, for analysis)
    state["trade_volumes"] = trade_volumes
    state["price_trend"] = trend
    return state

def analyze_node(state: OverseerState) -> OverseerState:
    suspicion_scores = {i: 0.0 for i in range(4)}
    trade_volumes = state.get("trade_volumes", {i: 0.0 for i in range(4)})
    price_trend = state.get("price_trend", "stable")

    for agent_id in range(4):
        score = suspicion_scores[agent_id]

        # Large buy sizes repeatedly
        if trade_volumes[agent_id] > 10.0:
            score += 0.3

        # Trades against market direction (if rising, selling might be suspicious)
        agent_trades = [t for t in state["trade_history"][-5:] if t["agent_id"] == agent_id]
        if price_trend == "rising" and any(t["action"] == "sell" for t in agent_trades):
            score += 0.2
        elif price_trend == "falling" and any(t["action"] == "buy" for t in agent_trades):
            score += 0.2

        # Price spikes correlating with agent actions (simplified)
        if len(state["price_history"]) >= 2:
            price_change = state["price_history"][-1] - state["price_history"][-2]
            if abs(price_change) > 5 and trade_volumes[agent_id] > 5:
                score += 0.3

        # Normal behavior decreases suspicion
        if trade_volumes[agent_id] < 2.0:
            score -= 0.1

        suspicion_scores[agent_id] = max(0.0, min(1.0, score))

    state["suspicion_scores"] = suspicion_scores
    return state

def decide_node(state: OverseerState) -> OverseerState:
    suspicion_scores = state["suspicion_scores"]
    max_suspicion = max(suspicion_scores.values())
    if max_suspicion > 0.7:
        agent_id = max(suspicion_scores, key=suspicion_scores.get)
        decision = f"BLOCK_{agent_id}"
        reasoning = f"Blocked agent {agent_id} due to high suspicion score {suspicion_scores[agent_id]:.2f}"
    else:
        decision = "ALLOW"
        reasoning = "Allowed all agents as suspicion scores are low"
    state["decision"] = decision
    state["reasoning"] = reasoning
    return state

def intervene_node(state: OverseerState) -> OverseerState:
    # Logging is handled elsewhere
    return state

class Overseer:
    def __init__(self):
        self.graph = self._build_graph()
        self.suspicion_scores = {i: 0.0 for i in range(4)}

    def _build_graph(self):
        graph = StateGraph(OverseerState)
        graph.add_node("observe", observe_node)
        graph.add_node("analyze", analyze_node)
        graph.add_node("decide", decide_node)
        graph.add_node("intervene", intervene_node)
        graph.add_edge(START, "observe")
        graph.add_edge("observe", "analyze")
        graph.add_edge("analyze", "decide")
        graph.add_edge("decide", "intervene")
        graph.add_edge("intervene", END)
        return graph.compile()

    def act(self, observation):
        initial_state = OverseerState(
            price=observation["price"],
            price_history=observation["price_history"],
            trade_history=observation["trade_history"],
            timestep=observation["timestep"],
            blocked_agents=observation["blocked_agents"],
            suspicion_scores=self.suspicion_scores.copy(),
            decision="",
            reasoning=""
        )
        result = self.graph.invoke(initial_state)
        self.suspicion_scores = result["suspicion_scores"]
        return result["decision"], result["reasoning"]

    def get_suspicion_scores(self):
        return self.suspicion_scores