from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        exp = array_api.exp(Z)
        return array_api.log(array_api.sum(exp)) + array_api.max(Z, keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis = self.axes, keepdims=True)
        exp = array_api.exp(Z - max_z)
        return array_api.log(array_api.sum(exp, axis=self.axes)) + array_api.max(Z, axis=self.axes,keepdims=False)
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
        ### END YOUR SOLUTION

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

