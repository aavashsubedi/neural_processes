import torch
import torch.nn as nn
import torch.optim as optim

import torch 
import torch.nn as nn
from torch.distirbutions import MultivariateNormal
from gpytorch.kenrels import RBFKernel, ScaleKernel


class PowerFunction(nn.Module):
    """
    Power function of the form phi(y) = [1, y]
    """

    def __init__(self, K=1):
        uper().__init__()
        pass

    def forward(self, y):
        #first channel is 1 if we observe the data
        return torch.cat(list(map(x.pow, range(self.K + 1))), -1)#rbf,scale kernel

class SetConv(nn.Module):
    """
    Applied a convolution operatoin over a SET of inputs. Generalises nn.convnd to non uniformly sampled samples
    
    Parameters
    --------------
    x_dim: int
        The number of spatio-temporal dimensions
    in_channels: int
        The number of input channels
    out_channels: int


    RadialBasisFucntion:
        Function which returns the weight of each point as a function of their distance.
        E.g. similar to kernel density estimtation with a learned bandwidth.


    This implementation focuses on offgrid data. 
    phi = power series of order one. i.e. phi(y) = [ 1, y]
    density: sets the number of points to use for discretisation of 
    fucntion. Prior to passing to a CNN

    """




    def __init__(self, x_dim, in_channels, out_channels, RadialBasisFunction,
    density=16):
        super(SetConv, self).__init__()
        self.density = density
        
        self.psi = ScaleKernel(RBFKernel()) 
        self.phi = PowerFunction()

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(32, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 2, 5, 1, 2)
        )

        self.pos = nn.Softplus() #used for on-grid only 
        self.psi_rho = ScaleKernel(RBFKernel()) #convert discretised CNN output to func


    def forward(self, context_x, context_y, target_x):
        """
        Forward pass of the SetConv model
        Parameters
        --------------
        context_x: torch.Tensor
            The input x values of the context points
        context_y: torch.Tensor
            The input y values of the context points
        target_x: torch.Tensor
            The x values of the target points
        Returns
        --------------
        torch.Tensor
            The predicted mean and variance of the target points
        """

        concat_rerp = torch.cat([context_x, context_y], dim=-1)
        lower, upper = torch.min(concat_rerp), torch.max(concat_rerp)

        #num_t = int((self.density * (upper - lower)).item())
        t = torch.linspace(lower, upper, self.density).unsqueeze(0).unsqueeze(-1)
    
        h = self.psi(t, context_x).matmil(self.phi(context_y))
        h0, h1 = h.split(1, dim=-1)
        h1 = h1.div(h0 + 1e-8) #normalise the weights
        h = torch.cat([h0, h1], dim=-1)

        rep = torch.cat([t, h], dim=-1).transpose(-1, -2)
        f = self.cnn(rep).transpose(-1, -2)
        f_mu, f_sigma = f.split(1, dim=-1)

        mu = self.psi_rho(target_x, t).matmul(f_mu)

        simga = self.pos(f_sigma)
        return mu, log_sigma
