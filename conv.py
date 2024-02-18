import torch
import torch.nn as nn
import numpy as np
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{func.__name__} took {elapsed:.6f} seconds to run.")
        return result
    return wrapper

@timer
def conv_reference(Z, weight):
    Z_torch = torch.tensor(Z).permute(0,3,1,2)
    W_torch = torch.tensor(weight).permute(3,2,0,1)
    out = nn.functional.conv2d(Z_torch, W_torch)
    return out.permute(0,2,3,1).contiguous().numpy()

@timer
def conv_naive(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    
    out = np.zeros((N,H-K+1,W-K+1,C_out));
    for n in range(N):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for y in range(H-K+1):
                    for x in range(W-K+1):
                        for i in range(K):
                            for j in range(K):
                                out[n,y,x,c_out] += Z[n,y+i,x+j,c_in] * weight[i,j,c_in,c_out]
    return out

@timer
def conv_matmul(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    out = np.zeros((N,H-K+1,W-K+1,C_out))
    
    for i in range(K):
        for j in range(K):
            out += Z[:,i:i+H-K+1,j:j+W-K+1,:] @ weight[i,j]
    return out
    
# Z = np.random.randn(10,32,32,8)
# W = np.random.randn(3,3,8,16)
# out = conv_reference(Z,W)
# out2 = conv_naive(Z,W)
# out3 = conv_matmul(Z,W)
# print(np.linalg.norm(out - out2))
# print(np.linalg.norm(out - out3))

# n = 6
# A = np.arange(n**2,dtype=np.float32).reshape(n,n)
# print(A)

# import ctypes
# print(np.frombuffer(ctypes.string_at(A.ctypes.data, A.nbytes), dtype=A.dtype, count=A.size))
# print(A.strides)

@timer
def conv_im2col(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    Ns,Hs,Ws,Cs = Z.strides
    hidden_dim = K*K*C_in
    A = np.lib.stride_tricks.as_strided(Z, shape=(N, H-K+1, W-K+1, K, K, C_in),
                                        strides=(Ns,Hs,Ws,Hs,Ws,Cs)).reshape(-1,hidden_dim)
    out = A @ weight.reshape(-1, C_out)
    return out.reshape(N, H-K+1, W-K+1, C_out)

Z = np.random.randn(100,32,32,8)
W = np.random.randn(3,3,8,16)
out = conv_reference(Z,W)
out2 = conv_im2col(Z,W)
out3 = conv_matmul(Z,W)
# out4 = conv_naive(Z,W)
print(np.linalg.norm(out - out2))
print(np.linalg.norm(out - out3))
# print(np.linalg.norm(out - out4))

