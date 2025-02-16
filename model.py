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
    def __init__(self, d_model: int, seq_len: int, dropout: float, n:float=10000.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len 
        self.dropout = nn.Dropout(p=dropout)
        
        enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1) # make it (seq_len, 1)
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

    def forward(self, q, k, v, mask): # actually q, k, v are the same and they all maps to some values in representation-wise
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

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block:MultiheadAttention, feed_forward_block:FeedForward, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block 
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block, cross_attention_block, feed_forward_block, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList):
        super().__init__() 
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self,
                 encoder:Encoder,decoder:Decoder,
                 src_embed:InputEmbedding, trg_embed:InputEmbedding, 
                 src_pos_enc:PosEncoding, trg_pos_enc:PosEncoding,
                 projection_layer:ProjectionLayer):
        super().__init__() 
        self.encoder = Encoder
        self.decoder = Decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed 
        self.src_pos_enc = src_pos_enc
        self.trg_pos_enc = trg_pos_enc
        self.projection_layer = projection_layer 
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_enc(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos_enc(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size:int, trg_vocab_size:int ,
                      src_seq:int, trg_seq:int,
                      d_model:int=512, N:int=6, h:int=8,
                      dropout:float=0.1, d_ff:int=2048) -> Transformer:
    src_embed = InputEmbedding(src_vocab_size, d_model)
    trg_embed = InputEmbedding(trg_vocab_size, d_model)
    
    src_pos_enc = PosEncoding(d_model, src_seq, dropout)
    trg_pos_enc = PosEncoding(d_model, trg_seq, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, h, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, trg_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos_enc, trg_pos_enc, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

if __name__ == '__main__':
    print('Welcome!')