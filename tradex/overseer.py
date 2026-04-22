import torch
import torch.nn as nn
import numpy as np

class Overseer(nn.Module):
    def __init__(self, obs_dim=54, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        self.actor_intervene = nn.Sequential(
            nn.Linear(hidden_dim // 2, 2)
        )
        self.actor_target = nn.Sequential(
            nn.Linear(hidden_dim // 2, 4)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        h = self.net(x)
        
        logits_intervene = self.actor_intervene(h)
        logits_target = self.actor_target(h)
        value = self.critic(h)
        
        threat_score = x[:, 53]
        
        # Add strong prior favoring intervention when threat is critical
        prior_boost = (threat_score > 0.85).float() * 10.0
        logits_intervene[:, 1] += prior_boost
        
        prob_intervene = torch.softmax(logits_intervene, dim=-1)
        prob_target = torch.softmax(logits_target, dim=-1)
        
        p_allow = prob_intervene[:, 0:1]
        p_block = prob_intervene[:, 1:2] * prob_target
        
        probs = torch.cat([p_allow, p_block], dim=-1)
        # Add epsilon to prevent log(0)
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        return probs, value
        
    def select_action(self, obs_vec, deterministic=False):
        x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        probs, val = self.forward(x)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            return action.item(), None, val, None, probs[0].detach().cpu().numpy()
            
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), val, dist.entropy(), probs[0].detach().cpu().numpy()

def encode_observation(obs):
    v = [
        obs['price']/100.0,
        obs['momentum'],
        obs['volatility'],
        np.log1p(obs['liquidity']/1000.0),
        obs['price_dev']/100.0,
        obs['timestep']/50.0,
        obs['stage']/5.0,
        obs.get('threat_score', 0.0) 
    ]
    v += [p/100.0 for p in obs['price_history']]
    
    for i in range(4):
        s = obs['stats'][i]
        v += [
            s['buys']/10.0,
            s['sells']/10.0,
            s['avg_size']/20.0,
            s['blocked']/10.0,
            s['burst_score']/50.0,
            s['pump_score']/100.0,
            s['dump_score']/100.0,
            s['coordination'],
            np.sign(s['pnl']) * np.log1p(abs(s['pnl']))
        ]
        
    return np.array(v, dtype=np.float32)

action_map = ["ALLOW", "BLOCK_0", "BLOCK_1", "BLOCK_2", "BLOCK_3"]