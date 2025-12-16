import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    shape=(fan_in,fan_out)
    a=gain*math.sqrt(6/(fan_in+fan_out))
    return rand(*shape,low=-a,high=a,**kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    shape=(fan_in,fan_out)
    std=gain*math.sqrt(2/(fan_in+fan_out))
    return randn(*shape,mean=0,std=std,**kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    shape=(fan_in,fan_out)
    gain=math.sqrt(2)
    a=gain*math.sqrt(3/fan_in)
    return rand(*shape,low=-a,high=a,**kwargs)
    ### END YOUR SOLUTION

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    shape=(fan_in,fan_out)
    gain=math.sqrt(2)
    std=gain*math.sqrt(1/fan_in)
    return randn(*shape,mean=0,std=std,**kwargs)
    ### END YOUR SOLUTION