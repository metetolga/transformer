import torch 
import numpy as np
import matplotlib.pyplot as plt

# TODO: readme
def get_pos_enc(seq_len, d, n=100):
    output = torch.empty((seq_len, d)) # make output as sequence length and d dimension
    
    for k in range(seq_len):
        for i in np.arange(d//2):
            denom = np.power(n, 2*i / d)
            output[k, 2*i] = np.sin(k / denom)
            output[k, 2*i+1] = np.cos(k / denom)

    return output

def get_pos_enc_vectorized(seq_len, d, n=1000):
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(-torch.arange(0, d, 2) * (np.log(n) * 2 / d))  # (d/2,)
    
    # Broadcasting: (seq_len, 1) / (1, d/2) -> (seq_len, d/2)
    pos_enc = torch.zeros((seq_len, d))
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    
    return pos_enc 

if __name__ == '__main__':
    print(get_pos_enc(4, 4)) 
    P = get_pos_enc_vectorized(seq_len=10000, d=5120, n=10000)
    cax = plt.matshow(P)
    plt.gcf().colorbar(cax)
    plt.show()

    seq_len = 10
    d = 10
    n = 1000
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  
    div_term = torch.exp(-torch.arange(0, d, 2) * (np.log(n) * 2 / d))
    

'''
P(k, 2i) = sin(k / n^(2i/d))
P(k, 2i+1) = cos(k / n^(2i/d))

k is index of the token, must be vectorized

1 / n^(2i/d) -> e^ln(1 / 2i/d) -> e^ln(d/2i) 
-> e^ln(d * 2i^(-1)) -> e^-ln(d * 2i) also e^ln(-d * 2i)


'''
    