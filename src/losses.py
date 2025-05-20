import numpy as np
from tensor import Tensor

def gather(t: Tensor, indices):
    """
    Gathers the log-probabilities for the correct class labels.
    t: Tensor of shape (N, C), indices: numpy array of shape (N,)
    """
    N = t.data.shape[0]
    gathered = t.data[np.arange(N), indices].reshape(N, 1)
    out = Tensor(gathered, (t,), 'gather')
    def _backward():
        grad = np.zeros_like(t.data)
        grad[np.arange(N), indices] = out.grad.reshape(-1)
        t.grad += grad
    out._backward = _backward
    return out

def cross_entropy_loss(logits: Tensor, targets):
    """
    Computes cross-entropy loss between logits and integer targets.
    logits: shape (N, C); targets: numpy array of shape (N,)
    """
    N = logits.data.shape[0]
    max_logits = np.max(logits.data, axis=1, keepdims=True)
    shifted = logits - Tensor(max_logits)
    exps = shifted.exp()
    sum_exps = exps.sum(axis=1, keepdims=True)
    log_sum_exps = sum_exps.log()
    log_probs = shifted - log_sum_exps
    target_log_probs = gather(log_probs, targets)
    loss = (target_log_probs * -1).sum() / N
    return loss