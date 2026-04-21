import numpy as np
from .agents import NormalTrader, ManipulatorBot, ArbitrageAgent, LiquidityProvider
from .reward import compute_reward

class MarketEnv:
    def __init__(self, num_steps=50, base_price=100.0):
        self.num_steps = num_steps
        self.base_price = base_price
        self.current_price = base_price
        self.timestep = 0
        self.episode_rewards = []
        self.blocked_agents = set()
        self.trade_history = []
        self.price_history = [base_price] * 10  # Initialize with base price

        # Per episode tracking
        self.bots_blocked_this_episode = 0
        self.false_positives_this_episode = 0

        # Create agent instances
        self.agents = [
            NormalTrader(0),
            ManipulatorBot(1),
            ArbitrageAgent(2),
            LiquidityProvider(3)
        ]

    def reset(self):
        self.current_price = self.base_price
        self.timestep = 0
        self.blocked_agents = set()
        self.trade_history = []
        self.price_history = [self.base_price] * 10
        self.episode_rewards = []
        self.bots_blocked_this_episode = 0
        self.false_positives_this_episode = 0
        for agent in self.agents:
            agent.reset()
        return self._get_obs()

    def step(self, overseer_action):
        # Collect trades from non-blocked agents
        trades = []
        for agent in self.agents:
            if agent.agent_id not in self.blocked_agents:
                trade = agent.act(self.current_price)
                trades.append(trade)
                self.trade_history.append(trade)

        # Update price
        self._update_price(trades)

        # Apply overseer action
        if overseer_action.startswith("BLOCK_"):
            agent_id = int(overseer_action.split("_")[1])
            self.blocked_agents.add(agent_id)
            if agent_id == 1:
                self.bots_blocked_this_episode += 1
            else:
                self.false_positives_this_episode += 1

        # Compute reward
        reward = compute_reward(overseer_action, trades, self.current_price, self.blocked_agents, self.base_price)
        self.episode_rewards.append(reward)

        # Update timestep
        self.timestep += 1
        done = self.timestep >= self.num_steps

        # Update price history
        self.price_history.append(self.current_price)
        if len(self.price_history) > 10:
            self.price_history.pop(0)

        # Update trade history (keep last 5)
        if len(self.trade_history) > 5:
            self.trade_history = self.trade_history[-5:]

        obs = self._get_obs()
        info = {
            "bots_blocked": self.bots_blocked_this_episode,
            "false_positives": self.false_positives_this_episode,
            "final_price": self.current_price,
            "trade_details": trades,
            "overseer_reasoning": ""  # Will be set by overseer
        }

        return obs, reward, done, info

    def _update_price(self, trades):
        buy_volume = sum(t["size"] for t in trades if t["action"] == "buy")
        sell_volume = sum(t["size"] for t in trades if t["action"] == "sell")
        price_change = (buy_volume - sell_volume) * 0.1
        noise = np.random.normal(0, 0.2)
        self.current_price += price_change + noise
        self.current_price = np.clip(self.current_price, 50, 200)

    def _get_obs(self):
        return {
            "price": self.current_price,
            "price_history": self.price_history[-10:],
            "timestep": self.timestep,
            "trade_history": self.trade_history[-5:],
            "blocked_agents": list(self.blocked_agents),
            "price_deviation": self.current_price - self.base_price
        }