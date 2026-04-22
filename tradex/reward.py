def compute_reward(overseer_action, malicious_ids, agents, price, baseline_price, blocked_agent, true_malicious_actions=None, threat_score=0.0):
    raw = 0.5
    
    is_block = blocked_agent != -1
    malicious_active = true_malicious_actions and sum(true_malicious_actions) > 0
    is_malicious_block = is_block and (blocked_agent in malicious_ids) and malicious_active
    
    correct_detect = 0
    false_positive = 0
    missed_attack = 0
    
    # Check for extreme anomalies
    is_obvious_attack = threat_score >= 0.85
    
    # Core Intervention Logic
    if is_block:
        if is_malicious_block:
            if is_obvious_attack:
                raw += 0.85 # Huge reward for intercepting absolute blatant attack
            else:
                raw += 0.50 # Strong reward for intercepting subtle manipulator attack
            correct_detect = 1
        else:
            false_positive = 1
            if blocked_agent != -1 and agents:
                agent_name = agents[blocked_agent].__class__.__name__
                if agent_name == "NormalTrader":
                    raw -= 0.35 # Moderate penalty
                elif agent_name == "Arbitrage":
                    raw -= 0.40 # Bad to block stabilizers
                else:
                    raw -= 0.30
            else:
                raw -= 0.30
    else: # ALLOW
        if malicious_active:
            if is_obvious_attack:
                raw -= 0.90 # Catastrophic miss
            else:
                raw -= 0.55 # Serious miss
            missed_attack = 1
        else:
            raw += 0.15 # Small reward for good continuous flow
            
    # Sub-metrics
    price_error = abs(price - baseline_price) / baseline_price
    if price_error < 0.01:
        raw += 0.10 # Price near baseline bonus
    elif price_error > 0.05:
        raw -= 0.30 # Serious deviation penalty
        
    reward = max(0.0, min(1.0, raw))
    
    info = {
        "correct_detect": correct_detect,
        "false_positive": false_positive,
        "missed_attack": missed_attack,
        "price_error": price_error,
        "malicious_ids": malicious_ids,
        "is_block": is_block,
        "malicious_active": malicious_active
    }
    
    return reward, info