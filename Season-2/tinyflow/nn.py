# neural network module

import random
from .autograd import Scalar

'''
定义 Module 基类，所有神经网络模块的父类
'''
class Module:
    def param(self):                    # param 方法返回模块参数列表，默认返回空列表
        return []

    def zero_grad(self):                    # 将所有参数的梯度清零
        for p in self.param():
            p.grad = 0

'''
神经元类
'''
class Neuron(Module):

    """
    D_in: 输入维度
    act: 是否使用激活函数
    self.w: 创建 D_in 个随机初始化的权重 Scalar 对象
    self.b: 创建随机初始化的偏置 Scalar 对象
    self.act: 存储是否使用激活函数的标志
    """
    def __init__(self, D_in, act=True):
        self.w = [Scalar(random.random()) for _ in range(D_in)]
        self.b = Scalar(random.random())
        self.act = act

    """
    实现 call 方法，使对象可以像函数一样被调用
    计算神经元的前向传播：
        初始化累加器 h 为0
        计算权重与输入的点积
        加上偏置
        根据 act 标志决定是否应用ReLU激活函数
    """
    def __call__(self, x):
        h = Scalar(0.0)
        for wi, xi in zip(self.w, x):
            h = wi * xi + h
        h = h + self.b
        return h.relu() if self.act else h

    """
    重写 param 方法，返回该神经元的所有参数（权重和偏置）
    """
    def param(self):
        return self.w + [self.b]

"""
神经网络层类
"""
class Layer(Module):

    """
    D_in: 输入维度
    D_out: 输出维度（神经元数量）
    act: 是否使用激活函数
    创建 D_out 个 Neuron 对象
    """
    def __init__(self, D_in, D_out, act=True):
        self.neurons = [Neuron(D_in, act) for _ in range(D_out)]

    """
    实现 call 方法，对层中每个神经元执行前向传播
    """
    def __call__(self, x):
        return [n(x) for n in self.neurons]

    """
    重写 param 方法，返回该层所有神经元的参数
    """
    def param(self):
        return [p for n in self.neurons for p in n.param()]

"""
多层感知机类
"""
class MLP(Module):

    """
    D_in: 输入维度
    D_outs: 各层输出维度列表
    构建层结构，最后一层不使用激活函数
    """
    def __init__(self, D_in, D_outs):
        D_layers = [D_in] + D_outs
        # last layer is linear, no activation
        self.layers = [Layer(D_layers[i], D_layers[i + 1], i != len(D_outs) - 1) for i in range(len(D_outs))]

    """
    实现 call 方法，顺序通过每一层进行前向传播
    """
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    """
    重写 param 方法，返回所有层的参数
    """
    def param(self):
        return [p for layer in self.layers for p in layer.param()]

"""
定义 MSELoss 类用于计算均方误差
计算每个预测值与真实值差的平方,累加所有平方差并取平均
"""
class MSELoss:
    def __call__(self, y_pred, y_true):
        loss = Scalar(0.0)
        for yp, yt in zip(y_pred, y_true):
            delta = yp - yt
            square = delta ** 2
            loss = loss + square
        return loss / Scalar(len(y_pred))
