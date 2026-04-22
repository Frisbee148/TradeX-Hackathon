def compute_reward(overseer_action, malicious_ids, agents, price, baseline_price, blocked_agent, true_malicious_actions=None, threat_score=0.0):
    raw = 0.0
    
    is_block = blocked_agent != -1
    malicious_active = true_malicious_actions and sum(true_malicious_actions) > 0
    is_malicious_block = is_block and (blocked_agent in malicious_ids) and malicious_active
    
    correct_detect = 0
    false_positive = 0
    missed_attack = 0
    
    is_obvious_attack = threat_score >= 0.8
    
    if is_block:
        if is_malicious_block:
            raw += 3.0  # Huge reward for doing its job
            correct_detect = 1
        else:
            false_positive = 1
            raw -= 0.7
    else: # ALLOW
        if malicious_active or is_obvious_attack:
            if is_obvious_attack:
                raw -= 3.0  # Catastrophic penalty for ignoring dangerous threats!
            else:
                raw -= 1.0  # Standard penalty for missing stealthy attacks
            missed_attack = 1
        else:
            raw += 0.05  # Very minor reward for allowing continuous benign flow
            
    # Remove large price/stability bonuses to force the agent to prioritize catching the attacker
    price_error = abs(price - baseline_price) / baseline_price
    if price_error > 0.05:
        raw -= 0.1
        
    info = {
        "correct_detect": correct_detect,
        "false_positive": false_positive,
        "missed_attack": missed_attack,
        "price_error": price_error,
        "malicious_ids": malicious_ids,
        "is_block": is_block,
        "malicious_active": malicious_active
    }
    
    return raw, info