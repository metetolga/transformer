import torch 
import numpy as np

def attention(query, key, val):
    # query, key and val is same, autoregressive
    ndims = query.size(-1)
    scaled_dot_prod = torch.matmul(query, key.T) / np.sqrt(ndims)

    # softmax
    e_scaled = torch.exp(scaled_dot_prod - scaled_dot_prod.max(dim=-1, keepdim=True).values)
    softmax = e_scaled / e_scaled.sum(dim=-1, keepdim=True)
    print('e_scaled', e_scaled)
    print('softmax', softmax)

    weights = torch.matmul(softmax, val)
    return weights 

if __name__ == '__main__':
    torch.manual_seed(42) # for reproducability
    ndims = 4
    queries = torch.rand((5,ndims)) # 5 words each of dim 4
    queries

    keys = queries.detach().clone() # first detach tensor from computation graph, then clone it 
    keys

    # every word has (4,) representation
    sentence = ['the', 'cat', 'sat', 'on', 'mat']

    # Q * K^T
    mat_mult = torch.matmul(queries, keys.T) # (5,4) * (4, 5) => (5,5) 
    mat_mult

    '''
    this part is different from the default dot-product attention
    default one has * sqrt(ndims), not / sqrt(ndims) 
    reason for division, large values of ndims leads to
    softmax of dimension multipied mat mult to exteremly small gradients
    '''
    # (Q * K^T) / sqrt(ndims)
    mat_mult_scaled = mat_mult / torch.sqrt(torch.tensor(ndims))
    mat_mult_scaled = mat_mult / np.sqrt(ndims) # instead of creating tensors in each call
    mat_mult_scaled
    torch.softmax(mat_mult_scaled, dim=1)

    # softmax[(Q * K^T) / sqrt(ndims)]
    rowwise_max = mat_mult_scaled.max(dim=-1, keepdim=True).values
    e_mat_mult_scaled = torch.exp(mat_mult_scaled - rowwise_max) 

    sum_e_mat_mult_scaled = torch.sum(e_mat_mult_scaled, dim=1, keepdim=True)
    # dim is the axis which will be collapsed on
    # dim=1 means collapsing the columns
    softmax = torch.divide(e_mat_mult_scaled, sum_e_mat_mult_scaled)

    values = queries.detach().clone() 
    values

    attention_weights = torch.matmul(softmax, values) # (5,5) * (5,4) = (5,4) same dimension size with queries
    print(f'attention weights: {attention_weights}') 

    torch.sum(torch.tensor([0.2471, 0.2690, 0.2150, 0.2304, 0.2452]))

    # quick recap of broadcasting 
    A = [[1,2,3], [4,5,6], [7,8,9]]
    sum_A = np.sum(A, axis=1, keepdims=True)
    A / sum_A 



'''
Softmax Function
1) [x y z] -> [ex ey ez]
2) sum = ex+ey+ez
3) arr / sum => [ex/sum ey/sum ez/sum] now its normalized 
def softmax(input_vector):
    # Calculate the exponent of each element in the input vector
    exponents = [exp(i) for i in input_vector]

    # Correct: divide the exponent of each value by the sum of the exponents
    # and round off to 3 decimal places
    sum_of_exponents = sum(exponents)
    probabilities = [round(exp(i) / sum_of_exponents, 3) for i in exponents]

    return probabilities
'''

'''
torch.sum(_, _, keepdim=True)
Why keepdim set to True?
since the input tensor is 2D. We should be getting 2D tensor as result, since this tensor will
be used for division operation.

For example we have 3,3 mat. 
A = tensor([[1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]])


we should be getting sum_A = [[6], [15], [24]] -> (3, 1)
softmax_A = A / sum_A

whenever division, every column divides its sum corresponding in dimensions. and every dimension 
in sum tensor corresponds to sum of each row.

# [[1/6, 2/6, 3/6],  # Rows sum to 1
#  [4/15, 5/15, 6/15],
#  [7/24, 8/24, 9/24]]

What happens with keepdims = False
A = tensor([[1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]])
sum_A = [6, 15, 24] -> (3) 
broadcasting (3) to (3,3)
prepend 1 if dims not equal, -> (1,3)
this would be doing column-wise division 
 
softmax_A = A / sum_A

'''

'''
broadcasting conditions
order the input array/tensor dimension in trailing.

begin from end till beginning.
1) shapes are both 1 OR
2) one of the shapes is 1 OR
3) one of the dimension is null OR
4) both shapes are equal

Finally, whenever broadcasting, choose the max of the corresponding dimensions

x.shape = (5, 3, 4, 1)
y.shape = ( , 3, 1, 1)
y -> broadcast firstly (1, 3, 1, 1) then choose max of the dimensions
(x+y).sum.shape = (5, 3, 4, 1)


x.shape = (5, 2, 4, 1)
y.shape = ( , 3, 1, 1)
can not broadcast, 2nd dimension mismatch
'''