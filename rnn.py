import needle
import torch 
import torch.nn as nn
import numpy as np

# model = nn.LSTMCell(20,100)

def sigmoid(x):
    return 1/(1+np.exp(-x))

# def lstm_cell(x, h, c, W_hh, W_ih, b):
    # i,f,g,o = np.split(W_hh@h + W_ih@x + b ,4)
    # i,f,g,o = sigmoid(i) , sigmoid(f), np.tanh(g), sigmoid(o)
    # c_out = f*c + i*g
    # h_out = o*np.tanh(c_out)
    # return h_out, c_out

# x = np.random.randn(1,20).astype(np.float32)
# h0 = np.random.randn(1,100).astype(np.float32)
# c0 = np.random.randn(1,100).astype(np.float32)

# h_, c_ = model(torch.Tensor(x), (torch.Tensor(h0), torch.Tensor(c0)))
# h, c = lstm_cell(x[0], h0[0], c0[0],
#                  model.weight_hh.detach().numpy(),
#                  model.weight_ih.detach().numpy(), 
#                  (model.bias_ih + model.bias_hh).detach().numpy())

# print(np.linalg.norm(h_.detach().numpy() - h), np.linalg.norm(c_.detach().numpy() - c))

model = nn.LSTM(20, 100, 1)

# X = np.random.randn(50,20).astype(np.float32)
# h0 = np.random.randn(1,100).astype(np.float32)
# c0 = np.random.randn(1,100).astype(np.float32)


# def lstm(X, h, c, W_hh, W_ih, b):
#     H = np.zeros((X.shape[0], h.shape[0]))
#     for t in range(X.shape[0]):
#         h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
#         H[t,:] = h
#     return H, c

# batching
def lstm_cell(x, h, c, W_hh, W_ih, b):
    i,f,g,o = np.split(h@W_hh.T + x@W_ih.T + b[None,:], 4, axis=1)
    i,f,g,o = sigmoid(i) , sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*c + i*g
    h_out = o*np.tanh(c_out)
    return h_out, c_out

def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t,:,:] = h
    return H, c

X = np.random.randn(50,128,20).astype(np.float32)
h0 = np.random.randn(1,128,100).astype(np.float32)
c0 = np.random.randn(1,128,100).astype(np.float32)

H_, (hn_, cn_) = model(torch.Tensor(X),
                       (torch.Tensor(h0)[None,:,:],
                        torch.Tensor(c0)[None,:,:]))

H, cn = lstm(X, h0[0], c0[0], 
             model.weight_hh_l0.detach().numpy(), 
             model.weight_ih_l0.detach().numpy(), 
             (model.bias_hh_l0 + model.bias_ih_l0).detach().numpy())

print(np.linalg.norm(H - H_.detach().numpy()),
      np.linalg.norm(cn - cn_[0].detach().numpy()))


# def train_lstm(X, h0, c0, parameters):
#     H = X
#     for i in range(len(W_hh)):
#         H, cn = lstm(X, h0[i], c0[i], parameters[i])
#     l = loss(H, Y)
#     l.backward()
#     opt.step()
