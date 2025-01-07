import torch 
import torch.nn as nn
from diffusion_base import Diffusion_Base
from utils import _extract_into_tensor


class NDP(Diffusion_Base):
    def __init__(self, cfg,
                noise_steps=1000,
                beta_start=1e-4, beta_end=0.02,
                img_size=64, device="cuda", bw=False,
                use_cosine=False):

        super(NDP, self).__init__(cfg, noise_steps=noise_steps,
                                  beta_start=beta_start, beta_end=beta_end,
                                  img_size=img_size, device=device, bw=bw,
                                  use_cosine=use_cosine)
    def forward(self, x, t):
        #to stick to the convention of NDP paper
        return self.noise_image(x, t)

    def ddpm_backward_step(self, noise, yt, t):

        #given a noise, yt and time t, we calculate the next noise
        alpha = _extract_into_tensor(self.alpha, t, yt.shape)
        alpha_hat = _extract_into_tensor(self.alpha_cumprod, t, yt.shape)
        beta = _extract_into_tensor(self.beta, t, yt.shape)

        mask = (t > 0) * torch.randn_like(yt)
        yt_minus_one = 1 / torch.sqrt(alpha) * (
                    yt - (( 1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise)
        yt_minus_one = yt_minus_one + torch.sqrt(1 - alpha) * mask

        return yt_minus_one

    def sample(self, x, mask, model, outptut_dim=1, n=10000):

        y_target = torch.randn_like((len(x), outptut_dim))

        if mask is None:
            mask = torch.zeros_like(x[:, 0])

        def iter_func(y, t):
            predicted_noise = model(x, t)
            y = self.ddpm_backward_step(predicted_noise, y, t)
            return y, None

        t = (torch.ones(n) * i).long().to(self.device)
        for i in reversed(range(1, self.noise_steps)):
            y_target, _ = iter_func(y_target, t)
        return y_target
 
    

    def conditional_sample(
        self, x, mask, *, x_context, y_context, 
        mask_context, model_fn, num_inner_steps: int=5, 
        method: str="repaint"
    ):

        if mask is None:
            mask = torch.zeros_like(x[:, 0])
        if mask_context is None:
            mask_context = torch.zeros_like(x_context[:, 0])

        x_augmented = torch.concatenate([x_context, x], dim=0)
        mask_augmented = torch.concatenate([mask_context, mask], dim=0)

        num_context = len(x_context)

        def repaint_inner(yt_target, t):

            yt_context = self.forward(y_context, t)[0]
            y_augmented = torch.concatenate([yt_context, yt_target], dim=0)
            noise_hat = model.forward(t, y_augmented, x_augmented, mask_augmented)
            y = self.ddpm_backward_step(noise_hat, y_augmented, t)
            y = y[num_context:]

            #one step forward from t-1 to t using repaint
            z = torch.randn_like(y)
            beta_t_minus_one = _extract_into_tensor(self.beta, t - 1, y.shape)
            y = torch.sqrt(1.0 - beta_t_minus_one) * y + torch.sqrt(beta_t_minus_one) * z
            return y
        
        def repaint_outer(y, t):
            for i in range(num_inner_steps):
                y = repaint_inner(y, t)

            #step backward t - > t-1
            yt_context = self.forward(y_context, t)[0]
            y_augmented = torch.concatenate([yt_context, y], dim=0)
            noise_hat = model(t, y_augmented, x_augmented, mask_augmented)
            y = self.ddpm_backward_step(noise_hat, y_augmented, t)
            y = y[num_context:]
            return y
        

        y_target = torch.randn_like(y_context)
        for i in reversed(range(1, self.noise_steps)):
            y_target = repaint_outer(y_target, i)
        return y_target