import torch
import torch.nn as nn
import numpy as np

class Overseer(nn.Module):
    def __init__(self, obs_dim=54, hidden_dim=256, out_dim=5):
        super().__init__()
        # Extracted feature processor
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
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, out_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        h = self.net(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value
        
    def select_action(self, obs_vec, deterministic=False):
        x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        logits, val = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        
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