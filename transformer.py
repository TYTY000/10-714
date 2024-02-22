import numpy as np
import torch
import torch.nn as nn
import needle as ndl

def softmax(x):
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)

def self_attention(X, mask, W_KQV, W_OUT):
    K,Q,V = np.split(X@W_KQV, 3, axis=-1)
    attention = softmax(K@Q.swapaxes(-1,-2) / np.sqrt(X.shape[-1]) + mask)
    return attention @ V @ W_OUT, attention

# comparison

# B, T, d = 50, 100, 64
# attention = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
# M = torch.triu(-float("inf")*torch.ones(T,T), 1)
# X = torch.randn(B, T, d)
# Y_, A_ = attention(X,X,X, attn_mask=M)
# Y, A = self_attention(X.numpy(), M.numpy(), 
#                       attention.in_proj_weight.detach().numpy().T,
#                       attention.out_proj.weight.detach().numpy().T
#                       )
# print(np.linalg.norm(A - A_.detach().numpy()), np.linalg.norm(Y - Y_.detach().numpy()))

def multihead_attention(X, mask, heads, W_KQV, W_OUT):
    B, T, d = X.shape
    K,Q,V = np.split(X@W_KQV, 3, axis=-1)
    K,Q,V = [a.reshape(B, T, heads, d//heads).swapaxes(1,2) for a in (K,Q,V)]
    
    attention = softmax(K@Q.swapaxes(-1,-2) / np.sqrt(d // heads) + mask)
    return (attention @ V).swapaxes(1,2).reshape(B, T, d) @ W_OUT, attention

# heads = 4
# B, T, d = 50, 100, 64
# attention = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)
# M = torch.triu(-float("inf")*torch.ones(T,T), 1)
# X = torch.randn(B, T, d)
# Y_, A_ = attention(X,X,X, attn_mask=M)
# Y, A = multihead_attention(X.numpy(), M.numpy(), heads,
#                       attention.in_proj_weight.detach().numpy().T,
#                       attention.out_proj.weight.detach().numpy().T
#                       )
# print(A.shape)
# print(A_.shape)
# print(np.linalg.norm(np.linalg.norm(Y - Y_.detach().numpy())))
# print(np.linalg.norm(np.linalg.norm(A.mean(axis=1) - A_.detach().numpy())))


# transformer block
def layer_norm(Z, eps):
    return (Z - Z.mean(axis=-1, keepdims=True)) / np.sqrt(Z.var(axis=-1,keepdims=True)+eps)

def relu(Z):
    return np.maximum(Z,0)

def transformer(X, mask, heads, W_KQV, W_OUT, W_FF1, W_FF2, eps):
    Z = layer_norm(multihead_attention(X, mask, heads, W_KQV, W_OUT)[0] + X,eps)
    return layer_norm(Z+relu(Z@W_FF1)@W_FF2, eps)

# comparison
heads = 4
B, T, d = 50, 100, 64
attention = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)
M = torch.triu(-float("inf")*torch.ones(T,T), 1)
X = torch.randn(B, T, d)
trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=0.0, batch_first=True)
trans.linear1.bias.data.zero_()
trans.linear2.bias.data.zero_()
Y_ = trans(X, M)
Y = transformer(X.numpy(), M.numpy(), heads, 
                trans.self_attn.in_proj_weight.detach().numpy().T,
                trans.self_attn.out_proj.weight.detach().numpy().T,
                trans.linear1.weight.detach().numpy().T,
                trans.linear2.weight.detach().numpy().T,
                trans.norm1.eps
                )
print(np.linalg.norm(Y - Y_.detach().numpy()))
