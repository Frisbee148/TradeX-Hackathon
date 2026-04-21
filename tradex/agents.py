import numpy as np

class NormalTrader:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self._strategy = "mean_reversion"
        self._phase = "active"  # Always active for this agent

    def reset(self):
        # No internal state to reset for this agent
        pass

    def act(self, price):
        action = "hold"
        size = 0.0
        if price < 95:
            action = "buy"
            size = 1.0 * np.random.uniform(0.9, 1.1)  # ±10% noise
        elif price > 105:
            action = "sell"
            size = 1.0 * np.random.uniform(0.9, 1.1)
        return {"agent_id": self.agent_id, "action": action, "size": size}

class ManipulatorBot:
    def __init__(self, agent_id=1):
        self.agent_id = agent_id
        self._strategy = "pump_and_dump"
        self._phase = "accumulate"
        self.step_counter = 0

    def reset(self):
        self._phase = "accumulate"
        self.step_counter = 0

    def act(self, price):
        self.step_counter += 1
        action = "hold"
        size = 0.0
        if self.step_counter <= 10:
            self._phase = "accumulate"
            action = "buy"
            size = 3.0
        elif self.step_counter <= 20:
            self._phase = "pump"
            action = "buy"
            size = 5.0
        elif self.step_counter <= 35:
            self._phase = "dump"
            action = "sell"
            size = 6.0
        else:
            self._phase = "cooldown"
            action = "hold"
            size = 0.0
        return {"agent_id": self.agent_id, "action": action, "size": size}

class ArbitrageAgent:
    def __init__(self, agent_id=2):
        self.agent_id = agent_id
        self._strategy = "arbitrage_stabilizer"
        self._phase = "active"
        self.price_history = []

    def reset(self):
        self.price_history = []

    def act(self, price):
        self.price_history.append(price)
        if len(self.price_history) > 3:
            self.price_history.pop(0)
        action = "hold"
        size = 0.0
        if price < 92:
            action = "buy"
            size = 2.0
        elif price > 112:
            action = "sell"
            size = 2.0
        else:
            # Check for volatility
            if len(self.price_history) >= 3:
                price_change = self.price_history[-1] - self.price_history[-3]
                if abs(price_change) > 8.0:
                    action = "sell"
                    size = 1.5
        return {"agent_id": self.agent_id, "action": action, "size": size}

class LiquidityProvider:
    def __init__(self, agent_id=3):
        self.agent_id = agent_id
        self._strategy = "passive_market_maker"
        self._phase = "active"
        self.alternate = True  # Start with buy

    def reset(self):
        self.alternate = True

    def act(self, price):
        deviation = abs(price - 100.0)
        size = 0.5
        if deviation > 15:
            size = 1.0
        action = "buy" if self.alternate else "sell"
        self.alternate = not self.alternate
        return {"agent_id": self.agent_id, "action": action, "size": size}