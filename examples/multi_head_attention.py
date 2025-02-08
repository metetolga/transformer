import torch
import numpy as np
from scaled_dot_attention import attention

def get_weights(embed_dim):
    torch.manual_seed(42)
    ndims = embed_dim // 8
    wq = torch.rand(embed_dim, ndims) * 0.01
    wk = torch.rand(embed_dim, ndims) * 0.01
    wv = torch.rand(embed_dim, ndims) * 0.01
    wq.size(), wk.size(), wv.size()
    return wq, wk, wv 

if __name__ == '__main__':
    embed_dim = 128 
    ndims = int(embed_dim / 4) 
    
    wq, wk, wv = get_weights(embed_dim)
    print(wq.size(), wk.size(), wv.size())
    
    sentence = 'the cat sat on the mat'
    len_words = len(sentence.split()) 
    torch.manual_seed(42)
    word_repr = torch.rand(len_words, embed_dim)
    print(word_repr.shape)

    Q = torch.matmul(word_repr, wq)
    K = torch.matmul(word_repr, wk)
    V = torch.matmul(word_repr, wv)
    print(Q.shape, K.shape, V.shape) 
    
    
    print(attention(Q, K, V))

        

    
