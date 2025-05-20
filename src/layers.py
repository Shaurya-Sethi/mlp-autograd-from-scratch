import numpy as np
from tensor import Tensor

def selu(x: Tensor):
    """
    SELU activation function.
    """
    scale = 1.0507
    alpha = 1.67326
    data = scale * np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1))
    out = Tensor(data, (x,), 'selu')
    def _backward():
        grad_input = np.where(x.data > 0, scale, scale * alpha * np.exp(x.data)) * out.grad
        x.grad += grad_input
    out._backward = _backward
    return out

def dropout(x: Tensor, p=0.3, training=True):
    """
    Dropout regularization.
    """
    if not training:
        return x
    mask = (np.random.rand(*x.data.shape) > p).astype(np.float32)
    data = x.data * mask / (1 - p)
    out = Tensor(data, (x,), 'dropout')
    def _backward():
        x.grad += out.grad * mask / (1 - p)
    out._backward = _backward
    return out

class Linear:
    """
    Fully connected linear layer.
    """
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nout, nin) * np.sqrt(2.0 / nin))
        self.b = Tensor(np.zeros((nout,)))
        self.params = [self.W, self.b]

    def __call__(self, x: Tensor):
        return x.matmul(self.W.transpose()) + self.b

class BatchNorm1d:
    """
    1D Batch Normalization.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = Tensor(np.ones((num_features,)))
        self.beta = Tensor(np.zeros((num_features,)))
        self.params = [self.gamma, self.beta]
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)
        self.training = True
        self.eps = eps
        self.momentum = momentum

    def __call__(self, x: Tensor):
        # x: (N, num_features)
        if self.training:
            mu_val = np.mean(x.data, axis=0)
            var_val = np.mean((x.data - mu_val) ** 2, axis=0)

            mu = Tensor(mu_val)
            var = Tensor(var_val)

            self.running_mean = self.momentum * mu_val + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var_val + (1 - self.momentum) * self.running_var

            x_norm = (x - mu) / ((var + Tensor(self.eps)).pow(0.5))
        else:
            mu = Tensor(self.running_mean)
            var = Tensor(self.running_var)
            x_norm = (x - mu) / ((var + Tensor(self.eps)).pow(0.5))

        out = self.gamma * x_norm + self.beta
        return out