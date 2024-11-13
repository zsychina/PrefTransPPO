import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim, actor=True):
        super(MLP, self).__init__()
        if actor:
            self.mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(), 
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),             
                nn.Linear(hidden_dim, out_dim),
                nn.Tanh(),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(), 
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),   
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )            
    
    def forward(self, x):
        # [batch, state_dim]
        x = self.mlp(x)
        return x

