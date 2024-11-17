import torch

import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, nhead, dim_lin=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        
        self.lin1 = nn.Linear(dim, dim_lin)
        self.lin2 = nn.Linear(dim_lin, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.self_attn(x, x, x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = F.relu(self.lin1(x))
        x2 = self.dropout1(x2)
        x2 = self.lin2(x2)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, L=6):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(L)])
        self.num_layers = L

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    encoder_layer = TransformerEncoderLayer(dim=512, nhead=8)
    transformer_encoder = TransformerEncoder(encoder_layer, L=6)
    x = torch.rand(10, 32, 512)
    out = transformer_encoder(x)
    print(out.shape)

