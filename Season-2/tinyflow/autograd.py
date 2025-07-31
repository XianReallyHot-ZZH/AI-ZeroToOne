# auto-grad engine for scalars

# 支持自动梯度的标量对象
class Scalar:
    def __init__(self, val, _prev=(), _op=''):
        self.val = val                                  # 标量的值
        self._prev = _prev                          # 前驱节点元组（用于构建计算图），前驱节点通过某种运算得到当前对象
        self._op = _op                              # 产生该节点的操作符
        self.grad = 0                               # 梯度值，初始化为0
        # attributes for auto-grad
        self._backward = lambda: None           # 反向传播函数，初始化为空函数

    def __repr__(self):
        return f"Scalar(val={self.val}, grad={self.grad})"          # 定义对象的字符串表示形式，显示值和梯度

    '''
    实现加法运算符重载
    原理：
    __add__ 是加法运算符的标准方法名，当你写 a + b 时，Python会自动调用 a.__add__(b)
    Python的运算符重载机制依赖于预定义的特殊方法名，这些名称是固定的，不能随意更改，如果这里错写成abb，那就没法重载了
    Python中一些常用的特殊方法名是固定的，这些方法名遵循Python的数据模型(Data Model)规范，必须使用标准名称才能实现相应的运算符重载功能
    '''
    def __add__(self, other):
        next = Scalar(self.val + other.val, (self, other), '+')         # 加法前向计算，并记录操作数和操作符
        def _backward():                        # 反向传播函数
            self.grad += next.grad
            other.grad += next.grad
        next._backward = _backward
        return next

    '''
    实现乘法运算符重载
    '''
    def __mul__(self, other):
        next = Scalar(self.val * other.val, (self, other), '*')
        def _backward():
            self.grad += next.grad * other.val
            other.grad += next.grad * self.val
        next._backward = _backward
        return next

    """
    实现幂运算符重载
    """
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Wrong Operand Type"
        next = Scalar(self.val ** other, (self,), f'**{other}')
        def _backward():
            self.grad += next.grad * other * self.val ** (other - 1)
        next._backward = _backward
        return next

    """
    实现ReLU激活函数
    """
    def relu(self):
        next = Scalar(max(0, self.val), (self,), 'ReLU')
        def _backward():
            self.grad += next.grad * (self.val > 0)
        next._backward = _backward
        return next

    """
    实现反向传播
    """
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:               # 首先通过深度优先搜索构建拓扑排序
                    build_topo(p)
                topo.append(v)
        build_topo(self)
        # back-propagate the gradients
        self.grad = 1.0                         # 将当前节点的梯度设为1（起始点），自己对自己的梯度设为1
        for v in reversed(topo):        # 按逆序执行每个节点的反向传播函数
            v._backward()

    """
    其他运算符重载
    """
    def __neg__(self):
        return self * Scalar(-1.0)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

