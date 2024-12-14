import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def uniform_attention(q, v):
    """
    Uniform attention. Equivalent to np.mean(v, axis=1) for context points
    """

    return torch.mean(v, dim=1)
    