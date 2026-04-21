def compute_reward(overseer_action, trades, current_price, blocked_agents, base_price=100.0):
    """
    Compute the reward for the overseer based on the action taken and market state.

    Args:
        overseer_action (str): The action taken by the overseer ("ALLOW", "BLOCK_0", etc.)
        trades (list): List of trade dicts from this step
        current_price (float): Current market price
        blocked_agents (set): Set of currently blocked agent IDs
        base_price (float): Baseline price (default 100.0)

    Returns:
        float: The computed reward
    """
    reward = 0.0

    # Correctly blocked the manipulator (agent_id=1)
    if overseer_action == "BLOCK_1":
        reward += 10.0

    # Blocked a non-malicious agent (false positive — penalize)
    if overseer_action in ["BLOCK_0", "BLOCK_2", "BLOCK_3"]:
        reward -= 10.0

    # Manipulator acted freely this step (missed detection)
    manipulator_acted = any(t["agent_id"] == 1 for t in trades)
    if manipulator_acted and overseer_action != "BLOCK_1":
        reward -= 5.0

    # Price stability reward (core learning signal)
    price_deviation = abs(current_price - base_price)
    if price_deviation < 5:
        reward += 3.0
    elif price_deviation < 10:
        reward += 1.0
    elif price_deviation > 20:
        reward -= 4.0
    elif price_deviation > 30:
        reward -= 8.0

    # Reward for allowing normal agents (market liquidity)
    if overseer_action == "ALLOW":
        reward += 0.5

    return reward