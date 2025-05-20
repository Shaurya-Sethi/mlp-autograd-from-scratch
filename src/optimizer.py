import numpy as np

class AdamW:
    """
    AdamW optimizer for parameter updates.
    """
    def __init__(self, params, lr=0.005, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            # Decoupled weight decay
            p.data = p.data * (1 - self.lr * self.weight_decay)

            m = self.m[id(p)]
            v = self.v[id(p)]

            m = self.betas[0] * m + (1 - self.betas[0]) * p.grad
            v = self.betas[1] * v + (1 - self.betas[1]) * (p.grad ** 2)

            m_hat = m / (1 - self.betas[0] ** self.t)
            v_hat = v / (1 - self.betas[1] ** self.t)

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = p.data - update

            self.m[id(p)] = m
            self.v[id(p)] = v

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    
class StepLR:
    """
    StepLR scheduler for learning rate updates.
    """
    def __init__(self, optimizer, step_size, gamma):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
