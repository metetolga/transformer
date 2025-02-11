import torch 
import torch.nn as nn
from torch import Tensor 

import numpy as np
import math

# TODO: add comments to every possible line as learn progressively
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int): 
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PosEncoding(nn.Module):
    def __init__(self, seq_len:int, d_model:int, dropout_rate:float, n:float=10000.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len 
        self.dropout = nn.Dropout(p=dropout_rate)
        
        enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(n)) / d_model)
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        enc = enc.unsqueeze(0)
        self.register_buffer('enc', enc)
    
    def forward(self, x):
        # add the positional encoding till the length of the sequence(=x->(batch,seq_len,d_model))
        # also register buffer ensure its not included in computation graph. 
        x = x + self.enc[:, :x.shape[1], :]
        return self.dropout(x) 

class LayerNormalization(nn.Module):
    def __init__(self, features:int, eps:float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
    
    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq, d_model) average over the last dim 
        std = x.std(dim=-1, keepdim=True) # (batch, seq, d_model) average over the last dim 
        return self.alpha * (x - mean) / (std + self.eps) - self.bias
    
class FeedForward(nn.Module):
    def __init__(self, d_model:int, hidden:int, dropout:float):
        super().__init__()
        self.first = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.second = nn.Linear(hidden, d_model)
    
    def forward(self, x):
        return self.second(self.dropout(torch.relu(self.first(x)))) 

class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, heads:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0, f"Inappropriate size: {d_model} % {heads} == 0"

        self.dropout = nn.Dropout(dropout)
        self.d_head = d_model // heads # dims for each head
        # (batch, seq, ndim)
        self.wq = nn.Linear(d_model, d_model, bias=False) # query 
        self.wk = nn.Linear(d_model, d_model, bias=False) # key
        self.wv = nn.Linear(d_model, d_model, bias=False) # value
        self.wo = nn.Linear(d_model, d_model, bias=False) # last linear
    
    @staticmethod
    def attention(query:Tensor, key:Tensor, value:Tensor, mask, dropout:nn.Dropout):
        d_k = query.shape[-1] # query.size(-1)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq, seq) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # (batch, seq, d_model) -> (batch, seq, d_model)
        qry = self.wq(q)
        key = self.wk(k)
        val = self.wv(v)
        
        # reshape it into multi-headed 
        # then transpose the seq_len and head count
        # (batch, seq, d_model) --> (batch, seq, h, d_k) --> (batch, h, seq, d_k)       
        qry = qry.view(qry.shape[0], qry.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        val = val.view(val.shape[0], val.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiheadAttention.attention(qry, key, val, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.wo(x) # feed it into output neural network

if __name__ == '__main__':
    print('Welcome!')