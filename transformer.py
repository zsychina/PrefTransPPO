import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalAttention(nn.Module):
    def __init__(self, n_head, dim, dropout=0.1, max_seqlen=64):
        super(CausalAttention, self).__init__()
        assert dim % n_head == 0, f'{dim=}应该是{n_head=}的整数倍'
        self.n_head = n_head
        self.dim = dim
        self.max_seqlen = max_seqlen
        self.attn_dropout = nn.Dropout(dropout)

        self.Wq = nn.Linear(self.dim, self.dim)
        self.Wk = nn.Linear(self.dim, self.dim)
        self.Wv = nn.Linear(self.dim, self.dim)
        self.Wo = nn.Linear(self.dim, self.dim)

        self.register_buffer("bias", torch.tril(torch.ones(self.max_seqlen, self.max_seqlen))
                                     .view(1, 1, self.max_seqlen, self.max_seqlen))

    def forward(self, x):
        # [1, 16, 512]
        B, T, C = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # -> [B, n_head, T, C // self.n_head]
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        # 进行casual mask
        att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        y = self.attn_dropout(F.softmax(att, dim=-1)) @ v
        y = y.reshape(B, T, C)
        return self.Wo(y)

        
class Block(nn.Module):
    def __init__(self, n_head, dim, dropout=0.1, max_seqlen=64):
        super(Block, self).__init__()
        self.n_head = n_head
        self.dim = dim
        self.max_seqlen = max_seqlen
        self.dropout = nn.Dropout(dropout)

        self.ln_1 = nn.LayerNorm(self.dim)
        self.attn = CausalAttention(n_head=self.n_head, dim=self.dim, dropout=dropout, max_seqlen=self.max_seqlen)
        self.ln_2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.dim, self.dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, n_layer, in_dim, out_dim, n_head, dim, dropout=0.1, max_seqlen=64):
        super(Transformer, self).__init__()
        self.n_head = n_head
        self.in_dim = in_dim
        self.dim = dim
        self.max_seqlen = max_seqlen
        self.n_layer = n_layer
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(self.max_seqlen, self.dim),
            dim_regularizer = nn.Linear(self.in_dim, self.dim),
            h = nn.ModuleList([Block(n_head=self.n_head, dim=self.dim, dropout=dropout, max_seqlen=self.max_seqlen) 
                              for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.dim)
        ))

        self.lm_head = nn.Linear(self.dim, self.out_dim)

        # Calculate number of parameters
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("Number of transformer parameters: %.2fM" % (n_params/1e6,))

    def forward(self, x):
        # x shape: [batch, seqlen, feature]
        B, T, C = x.shape
        device = x.device
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # Project input features to model dimension
        # x = self.transformer.dim_regularizer(x.view(-1, C)).view(B, T, -1)  # [batch, seqlen, dim]
        x = self.transformer.dim_regularizer(x)  # [batch, seqlen, dim]
        
        # Add positional embeddings
        pos_emb = self.transformer.wpe(pos)  # [batch, seqlen, dim]
        x = x + pos_emb
        
        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Apply final layer norm
        x = self.transformer.ln_f(x)
        
        # Project to output dimension
        return self.lm_head(x)  # [batch, seqlen, out_dim]


if __name__ == '__main__':
    transformer = Transformer(n_layer=1, in_dim=119, out_dim=1, n_head=8, dim=128, max_seqlen=int(2e3))
    
    traj1_states_actions = torch.rand(3, 42, 119)
    
    # print(transformer(traj1_states_actions))
    print(transformer(traj1_states_actions).shape)
    
    


