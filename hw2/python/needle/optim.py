"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            beta=self.momentum
            grad=param.grad.data+self.weight_decay*param.data
            ut=self.u.get(param,0)
            new_u=(beta*ut+(1-beta)*grad)
            self.u[param]=new_u
            param.data=param.data-self.lr*self.u[param]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t+=1 # 很妙的，每执行一个step, +=1
        for param in self.params:
            if param.grad:
                grad=param.grad.detach().data+self.weight_decay*param.detach().data
                # 这两行也是参考的
                if param not in self.m.keys():
                    self.m[param] = ndl.zeros_like(param.grad, requires_grad=False)
                if param not in self.v.keys():
                    self.v[param] = ndl.zeros_like(param.grad, requires_grad=False)
                mt=self.m[param]
                vt=self.v[param]
                new_m=self.beta1*mt+(1-self.beta1)*grad.data
                self.m[param]=new_m
                new_v=self.beta2*vt+(1-self.beta2)*grad.data*grad.data
                self.v[param]=new_v
                m_bias=new_m/(1-self.beta1**self.t)
                v_bias=new_v/(1-self.beta2**self.t)
                param.data=param.detach().data-(self.lr*m_bias)/(ndl.power_scalar(v_bias,0.5).detach().data+self.eps)
        ### END YOUR SOLUTION
