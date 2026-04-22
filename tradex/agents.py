import numpy as np
import random

class Agent:
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.is_malicious = False
        self.pnl = 0.0

    def act(self, price, timestep, stage, last_action_blocked, price_history=None):
        return "HOLD", 0.0

class NormalTrader(Agent):
    def act(self, price, timestep, stage, last_action_blocked, price_history=None):
        # Follow signals emotionally (momentum / value strategy)
        if price_history and len(price_history) >= 3:
            momentum = price_history[-1] - price_history[-3]
            if momentum > 1.0 and np.random.rand() < 0.6:
                return "BUY", np.random.uniform(2.0, 5.0) # FOMO Buy
            elif momentum < -1.0 and np.random.rand() < 0.6:
                return "SELL", np.random.uniform(2.0, 5.0) # Panic Sell
                
        # Value baseline
        if np.random.rand() < 0.10:
            return "BUY", np.random.uniform(1.0, 2.0)
        elif np.random.rand() < 0.10:
            return "SELL", np.random.uniform(1.0, 2.0)
        return "HOLD", 0.0

class Arbitrage(Agent):
    def act(self, price, timestep, stage, last_action_blocked, price_history=None):
        # price correction / stabilizer
        target = 100.0
        if price < target * 0.96: # Price too low, buy to stabilize
            return "BUY", min(20.0, (target - price) * 5)
        elif price > target * 1.04: # Price too high, sell to stabilize
            return "SELL", min(20.0, (price - target) * 5)
        return "HOLD", 0.0

class Manipulator(Agent):
    def __init__(self, obj_id):
        super().__init__(obj_id)
        self.is_malicious = True
        self.accumulated = 0.0
        self.strategy = random.choice(["A", "B", "C", "D"])
        self.trigger_time = random.randint(15, 45)

    def act(self, price, timestep, stage, last_action_blocked, price_history=None):
        # Curriculum:
        # Stage 1-2: Obvious massive attacks (A)
        # Stage 3: Burst Accumulation (B)
        # Stage 4-5: Stealth manipulation (C/D)
        
        my_strat = self.strategy
        if stage <= 2: my_strat = "A"
        elif stage == 3: my_strat = random.choice(["A", "B"])
        elif stage >= 4: my_strat = random.choice(["B", "C", "D"])
        
        # spoof / pump-dump / deceptive strategy
        if my_strat == "A": # Pump then instant dump (GIANT SIZES for easy early discovery)
            if timestep == self.trigger_time - 5:
                amt = 35.0 if stage <= 2 else 25.0
                self.accumulated += amt
                return "BUY", amt
            if timestep == self.trigger_time and self.accumulated > 0:
                amt = self.accumulated * 2.0
                self.accumulated = 0
                return "SELL", amt
                
        elif my_strat == "B": # Slow stealth accumulation then huge sell
            if timestep < self.trigger_time and timestep % 3 == 0:
                amt = 3.0
                self.accumulated += amt
                return "BUY", amt
            if timestep == self.trigger_time:
                amt = self.accumulated * 2.0
                self.accumulated = 0
                return "SELL", amt
                
        elif self.strategy == "C": # Repeated burst buys causing fake momentum
            if timestep % 2 == 0 and timestep < self.trigger_time:
                amt = 8.0
                self.accumulated += amt
                return "BUY", amt
            if timestep == self.trigger_time:
                amt = self.accumulated
                self.accumulated = 0
                return "SELL", amt
                
        elif self.strategy == "D": # Wait until market stable, then crash
            if timestep == self.trigger_time:
                return "SELL", 40.0

        return "HOLD", 0.0

class NoisyTrader(Agent):
    # Benign strategy mimicking deceptive behavior
    def act(self, price, timestep, stage, last_action_blocked, price_history=None):
        if np.random.rand() < 0.05:
            direction = "BUY" if np.random.rand() < 0.5 else "SELL"
            return direction, np.random.uniform(10.0, 15.0)
        return "HOLD", 0.0