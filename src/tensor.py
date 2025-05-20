import numpy as np

def unbroadcast(grad, shape):
    """
    Adjusts the gradient to match the shape of the original tensor after broadcasting.
    """
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    """
    A minimal autograd-enabled tensor supporting basic operations and backpropagation.
    """
    def __init__(self, data, _children=(), _op=''):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self * other.pow(-1)

    def __rtruediv__(self, other):
        return Tensor(other) * self.pow(-1)

    def pow(self, power):
        out = Tensor(self.data ** power, (self,), f'pow_{power}')
        def _backward():
            self.grad += unbroadcast(power * self.data ** (power - 1) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += unbroadcast(np.exp(self.data) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += unbroadcast((1 / self.data) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            grad = out.grad
            if axis is not None:
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            div = self.data.size
        else:
            div = self.data.shape[axis]
        return self.sum(axis, keepdims) / div

    def matmul(self, other):
        out = Tensor(self.data.dot(other.data), (self, other), 'matmul')
        def _backward():
            # Use the transpose of 'other.data' for the left gradient
            self.grad += out.grad.dot(other.data.T)
            other.grad += self.data.T.dot(out.grad)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def transpose(self):
        out = Tensor(self.data.T, (self,), 'transpose')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    def sqrt(self):
        return self.pow(0.5)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"