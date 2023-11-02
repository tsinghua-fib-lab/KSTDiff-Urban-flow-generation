import torch
import torch.nn as nn
from sublayers import MultiHeadAttention, FeedForward, Norm


class Building_Block(nn.Module):
    def __init__(self, edim, num_heads, dropout):
        super(Building_Block, self).__init__()

        self.embed_dim = edim

        self.attn = MultiHeadAttention(num_heads, self.embed_dim, dropout)

        self.pos_ffn = FeedForward(self.embed_dim, dropout=dropout)

        self.norm_1 = Norm(self.embed_dim)
        self.norm_2 = Norm(self.embed_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x): 
        x2 = self.norm_1(x)
        new_x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(new_x)
        new_x = new_x + self.dropout_2(self.pos_ffn(x2))
        return new_x
