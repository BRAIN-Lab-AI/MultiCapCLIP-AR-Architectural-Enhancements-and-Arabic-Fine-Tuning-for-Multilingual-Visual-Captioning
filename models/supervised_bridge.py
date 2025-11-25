import torch
import torch.nn as nn
import torch.nn.functional as F

#    Transformer Encoder Layer
# ============================================================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-Attention
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(src2))

        # Feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


#              Supervised Attention Bridge
# ============================================================

class SupervisedBridge(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.clip_dim = cfg["clip_dim"]
        self.mbart_dim = cfg["mbart_dim"]
        self.num_tokens = cfg["num_bridge_tokens"]
        self.n_layers = cfg["bridge_layers"]
        self.n_heads = cfg["bridge_heads"]
        self.ff_dim = cfg["bridge_ff_dim"]
        self.dropout = cfg["dropout"]

# 1) Linear projection to mBART space
        self.input_proj = nn.Linear(self.clip_dim, self.mbart_dim)

# 2) Transformer Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.mbart_dim,
                nhead=self.n_heads,
                dim_feedforward=self.ff_dim,
                dropout=self.dropout
            )
            for _ in range(self.n_layers)
        ])

# 3) Learnable query tokens for cross-attention pooling
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_tokens, self.mbart_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            self.mbart_dim,
            self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )

        self.output_norm = nn.LayerNorm(self.mbart_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

    def forward(self, clip_patch_feats):


        x = self.input_proj(clip_patch_feats)    # [B, N, mbart_dim]

        # Transformer encoder stack
        for layer in self.layers:
            x = layer(x)

        # learnable queries (32 tokens)
        B = x.size(0)
        queries = self.query_tokens.expand(B, -1, -1)

        pooled, _ = self.cross_attn(queries, x, x)

        return self.output_norm(pooled)
