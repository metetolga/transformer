import torch 
import torch.nn as nn
import math

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

if __name__ == '__main__':
    pass