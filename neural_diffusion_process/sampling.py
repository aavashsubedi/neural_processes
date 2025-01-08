import torch
import torch.nn as nn


def sample_prior(diffusion_process, model, x_grid):

    y0 = diffusion.sample(x_grid, mask=None, model=ema_model)
    return x_grid, y0

def sample_conditional(diffusion_process, model, x_grid):
    #model = ema model here
    x = torch.linspace(-2, 2, 57)[:, None]
    xc = torch.tensor([-1.0, 0.0, 1.0]).view(-1, 1)
    yc = torch.tensor([0.0, 1.0, 0.0]).view(-1, 1)

    y0 = diffusion_process.conditional_sample(
        x, mask=None, x_context=xc, y_context=yc,
        mask_context=None, model_fn=model
    )
    return x, y0, xc, yc


def sample_n_conditionals(x_test, y_test, x_context,
                 y_context, mask_context,
                diffusion_process, model, n=10):
    
    samples = diffusion_process.conditional_sample(
        x=x_test, mask=None, 
        x_context=x_context, y_context=y_context,
        mask_context=mask_context, model_fn=model)
    return samples
