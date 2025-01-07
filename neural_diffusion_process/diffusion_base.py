import torch 
import torch.nn as nn
from tqdm import tqdm 
import math
from utils import _extract_into_tensor

def cosine_schedule(beta_start=3e-4, beta_end=0.5, noise_steps=1000, s=0.008, **kwargs):
    t = torch.linspace(0, noise_steps, steps=noise_steps + 1)
    ft = torch.cos(((t/noise_steps) + s)/ (1+s) * math.pi * 0.5 )**2
    alpha_cumprod = ft /ft[0]
    betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
    betas = torch.clamp(betas, min=0.0001, max=0.9999)
    betas = (betas - betas.min()) /(betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start

class Diffusion_Base(nn.Module):
    """
    Prior q:
    
    posterior q:
    N(mu-tilda, beta_tilda)

    p_theta:
    N(mu_hat, sigma)
    """
    def __init__(self, cfg, noise_steps=1000,
                 beta_start=1e-4, beta_end=0.02,
                 img_size=64, device="cuda", bw=False, 
                 use_cosine=False):

        self.cfg = cfg
        
        self.noise_steps = self.cfg.noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = self.cfg.img_size
        self.device = device
        self.use_cosine = self.cfg.use_cosine
        self.bw = self.cfg.bw    
        super(Diffusion_Base, self).__init__()
        self.prepare_coefficents()

    def prepare_coefficents(self):

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0],).to(self.device),
        self.alpha_cumprod[:-1]], dim=0) #alpha_hat[t - 1]
        self.alpha_cumprod_next = torch.cat([self.alpha_cumprod[1:],
             torch.tensor([0.0]).to(self.device)], dim=0) #alpha_cumprod[t + 1]
        
        #calculations for diffusion q(x_t | x_(t -1))
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alpha_cumprod)
        self.sqrt_recip_alphas_cumprod = 1 / self.sqrt_alpha_cumprod
        self.sqrt_recip1_alphas_cumprod = torch.sqrt(1.0 / self.alpha_cumprod - 1.0)

        #calculations for posterior q(x_0 | x_t)

        #same as posterior variance
        self.beta_tilda = self.posterior_variance = (1 - self.alpha_cumprod_prev) * self.beta / (1 - self.alpha_cumprod)
        #check if any nan
        assert torch.isnan(self.beta_tilda).sum() == 0
        #log calculation clipped because the posterior variance is 0 at the first step
        #so we copy the second value to the first value
        self.posterior_log_variance_clipped = torch.log(torch.cat(
            [self.posterior_variance[1].unsqueeze(-1), self.posterior_variance[1:]], dim=0))
    
        self.posterior_mean_coeff_one = self.beta * (torch.sqrt(self.alpha_cumprod_prev ))/(1 - self.alpha_cumprod)

        self.posterior_mean_coeff_two = (torch.sqrt(self.alpha) * (1 - self.alpha_cumprod_prev)/ (1 - self.alpha_cumprod)) 
    def prepare_noise_schedule(self, s=0.008, max_beta=0.999,
                use_cosine=False):
        #linear noise schedule
        if not use_cosine:
            return torch.linspace(self.beta_start, self.beta_end,
                                  self.noise_steps)
    
        #logic taken from openai code.
        # Define the cumulative schedule alpha_hat (cosine)
        # t = torch.linspace(0, 1, steps=self.noise_steps + 1)
        # alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        # alpha_bar_t1, alpha_bar_t2 = alpha_bar[:-1], alpha_bar[1:]  # Corresponding alpha_bar values
        # betas = 1 - alpha_bar_t2 / alpha_bar_t1
        # betas = torch.clamp(betas, max=max_beta)

        betas = cosine_schedule(beta_start=self.beta_start, 
                beta_end=self.cfg.beta_end,
                         noise_steps=self.noise_steps, s=s)
    
        return betas.to(self.device)

    def noise_image(self, x, t):

        sqrt_alpha_hat = _extract_into_tensor(self.sqrt_alpha_cumprod, t, x.shape)
        sqrt_one_minus_alpha_hat = _extract_into_tensor(self.sqrt_one_minus_alpha_cumprod, t, x.shape)
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def q_sample(self, x_0, t):
        return self.noise_image(x=x_0, t=t)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):

        model.eval()

        with torch.no_grad():
            if self.bw:
                x = torch.randn((n, 1, self.img_size,
                                 self.img_size)).to(self.device)
            else:
                x = torch.randn((n, 3, self.img_size,
                                self.img_size)).to(self.device)

            for i in reversed(range(1, self.noise_steps)):
                #create tensor of length n
                t = (torch.ones(n) * i).long().to(self.device)
                model_output = model(x, t) #for classifier free guidance you also add the label here. Conditioned.
                
                if self.cfg.trained_variance:
                    predicted_noise, _ = torch.chunk(model_output, 2, dim=1)
                else:
                    predicted_noise = model_output
                
                # alpha = self.alpha[t][:, None, None, None]
                # alpha_hat = self.alpha_cumprod[t][:, None, None, None]
                # beta = self.beta[t][:, None, None, None] #beta shape: [b, 1, 1, 1]
                alpha = _extract_into_tensor(self.alpha, t, x.shape)
                alpha_hat = _extract_into_tensor(self.alpha_cumprod, t, x.shape)
                beta = _extract_into_tensor(self.beta, t, x.shape)

                if i > 1:
                    noise = torch.randn_like(x) #noise shape: [b, 1, 64, 64] for bw
                else:
                    noise = torch.zeros_like(x)
                #here beta is the variance at any given point that we specify through a scheduler
                x = 1 / torch.sqrt(alpha) * (
                    x - (( 1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) 
                x = x + torch.sqrt(beta) * noise
                #x_shape = [b, 1, 64, 64]

            model.train()
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255)
            return x