import math
import torch
#this kernel function computes the kernel matrix for the data
#ised to compute the covariance matrix kernel between x1, x2
from gpytorch.kernels import RBFKernel, ScaleKernel
def eq_kernel(x, length_scale, output_scale, jitter=1e-8):
    #we emplot thjis kernel quite alot
    x1 = x.unsqueeze(0)
    x2 = x.unsqueeze(1)

    pass

def rbfkernel(x, length_scale):

    k = exp(-0.5 & (x1 - x2) ** 2 / length_scale ** 2)
    pass