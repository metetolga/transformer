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
    def __init__(self, seq_len:int, d_model:int, dropout_rate:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len 
        self.dropout = nn.Dropout(p=dropout_rate)
        
        enc = torch.zeros(seq_len, d_model)
        # TO BE CONTINUED

if __name__ == '__main__':
    mbed = InputEmbedding(64, 512)
    mbed()