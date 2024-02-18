from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        exp = array_api.exp(Z)
        return array_api.log(array_api.sum(exp))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        Z_log = log(Z)
        # Compute the softmax of Z
        Z_softmax = exp(Z_log)
        # Compute the gradient of Z_log with respect to Z
        Z_log_grad = out_grad - array_api.summation(out_grad) * Z_softmax
        # Return the gradient of Z with respect to Z
        return Z_log_grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes=(axes,)

        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis = self.axes, keepdims=True)
        exp = array_api.exp(Z - max_z.broadcast_to(Z.shape))
        return array_api.log(array_api.sum(exp, axis=self.axes)) + Z.max(axis=self.axes,keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # i != k: gradient = exp(zi) / sum(exp(zk))
        # i == k: gradient = exp(zi) / sum(exp(zk))

        if self.axes:
            # get new shape
            j = 0
            shape = [1] * len(node.inputs[0].shape)
            for i in range(len(shape)):
                if i not in self.axes:
                    shape[i] = node.shape[j]
                    j+=1
            tmp_node = node.reshape(shape)
            tmp_grad = out_grad.reshape(shape)
        else:
            tmp_grad = out_grad
            tmp_node = node
        return tmp_grad * exp(node.inputs[0] - tmp_node)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

