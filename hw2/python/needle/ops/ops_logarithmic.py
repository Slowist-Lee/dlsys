from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        z_max_org=array_api.max(Z,1)
        z_max=array_api.expand_dims(z_max_org,1)
        Z_reduced=Z-z_max
        logsumexp=z_max_org+array_api.log(
            array_api.sum(array_api.exp(Z_reduced),1)
        )

        # 这里为什么要一个reshape，而不是Z reshape过去，真的没懂
        logsumexp_reshaped = array_api.expand_dims(logsumexp, 1)
        return Z-logsumexp_reshaped
        ### END YOUR SOLUTION
    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0] # shape (N, C)
        softmax = exp(node)  # softmax(z), shape (N, C)
        
        # print(f"Z: {Z}")
        # print(f"Z shape: {Z.shape}")
        sum_out = summation(out_grad, axes=(1,))            # shape (N,)
        sum_out = reshape(sum_out, (Z.shape[0], 1)).broadcast_to(Z.shape)
        return out_grad - sum_out * softmax


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes:
            axes = self.axes
        else:
            axes = tuple(range(len(Z.shape)))
        
        z_max_org=array_api.max(Z,axes)
        z_max=array_api.expand_dims(z_max_org,axis=axes)
        Z_reduced=Z-z_max
        return z_max_org + array_api.log(array_api.sum(array_api.exp(Z_reduced),axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z=node.inputs[0]
        # print(f"Z: {Z}")
        # 先补全 node 里被压缩的那个轴
        if self.axes:
            new_shape=[Z.shape[i] if i not in self.axes else 1 for i in range(len(Z.shape))]
        else:
            new_shape=[1 for _ in range(len(Z.shape))]

        # 这里的out_grad的维度也需要和Z.shape做一次匹配

        return out_grad.reshape(tuple(new_shape)).broadcast_to(Z.shape)*exp(Z-node.reshape(tuple(new_shape)).broadcast_to(Z.shape))

        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)