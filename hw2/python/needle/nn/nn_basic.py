"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.kaiming_uniform(in_features,out_features))
        if bias==True:
            # 这里需要考虑维度对齐的问题
            self.bias=Parameter(init.kaiming_uniform(fan_in=out_features,fan_out=1).reshape((1,out_features)))
        else:
            self.bias=None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        i = 0
        while X.shape[i]==1:
            i+=1
        # shape=(X.shape[i],self.out_features)
        if self.bias:
            # 不用转置
            # weight本来就是(in_features, out_features)
            # DEBUG
            # print(f"X. shape: {X.shape}")
            # print(f"weight.shape: {self.weight. shape}")
            # print(f"bias.shape: {self.bias.shape if self.bias else None}")
            # print(f"target shape: {shape}")
            # print(f"broadcast bias: {ops.broadcast_to(self.bias,shape)}")
            # print(f"broadcast bias shape: {ops.broadcast_to(self.bias,shape).shape}")
            # print(f"matmul: {X@self.weight}")
            # print(f"matmul shape: {(X@self.weight).shape}")
            out = X@self.weight
            return out+ops.broadcast_to(self.bias,out.shape)
        else:
            return X@ops.transpose(self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n=X.shape[0]
        return X.reshape((n,-1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            y=module(x)
            x=y
        return y
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n=logits.shape[1]
        # 一个要点是logsumexp而不是softmax
        # 另一个是要mean，真无敌了
        zy=ops.summation(logits*init.one_hot(n,y), axes=(1,))
        return ops.summation(ops.logsumexp(logits,axes=(1,))-zy)/logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias   = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean=Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_var= Parameter(init.ones(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            n=x.shape[0]
            x_mean=ops.summation(x,0)/n
            delta_x=x-ops.broadcast_to(x_mean.reshape((1,x.shape[1])),x.shape)
            x_var=ops.summation(ops.power_scalar(delta_x,2),0)/n
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*x_mean
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*x_var
            denomitor=ops.broadcast_to(ops.power_scalar((x_var+self.eps),0.5).reshape((1,x.shape[1])),x.shape)
            return self.weight*(delta_x/denomitor)+self.bias
        else:
            n=x.shape[0]
            delta_x=x-ops.broadcast_to(self.running_mean.reshape((1,x.shape[1])),x.shape)
            denomitor=ops.broadcast_to(ops.power_scalar((self.running_var+self.eps),0.5).reshape((1,x.shape[1])),x.shape)
            return self.weight*(delta_x/denomitor)+self.bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    # LayerNorm：在Layer维度上算平均，不是在batch上
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # self.weight=1
        # self.bias=0

        # 需要更新的参数!
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias   = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m=x.shape[1]
        x_mean=ops.summation(x,1)/m
        delta_x=x-ops.broadcast_to(x_mean.reshape((x.shape[0],1)),x.shape)
        x_var=ops.summation(ops.power_scalar(delta_x,2),1)/m
        denomitor=ops.broadcast_to(ops.power_scalar((x_var+self.eps),0.5).reshape((x.shape[0],1)),x.shape)
        return self.weight*(delta_x/denomitor)+self.bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        shape=x.shape
        mask=init.randb(*shape,p=1-self.p)
        return mask*x/(1-self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION
