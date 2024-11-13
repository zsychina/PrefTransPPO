import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


if __name__ == "__main__":
    mlp = MLP(in_dim=23, out_dim=1, hidden_dim=256)
    x = torch.rand(32, 128, 23)
    y = mlp(x[-1, :, :])
    print(y.shape)

