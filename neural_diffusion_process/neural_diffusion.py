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
        z = torch.randn_like(yt)
        mask = (t > 0).unsqueeze(-1).unsqueeze(-1) * z
        assert mask.shape == yt.shape
        yt_minus_one = 1 / torch.sqrt(alpha) * (
                    yt - (( 1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise)
        yt_minus_one = yt_minus_one + torch.sqrt(1 - alpha) * mask

        return yt_minus_one

    def sample(self, x, mask, model, output_dim=1, n=10000):
        y_target = torch.randn(len(x), output_dim)

        if mask is None:
            mask = torch.zeros_like(x[:, 0])

        def iter_func(y, t):
            predicted_noise = model(x, y, t, mask=mask)
            y = self.ddpm_backward_step(predicted_noise, y, t)
            return y, None

        t = torch.linspace(1, self.noise_steps, self.noise_steps)
        for i in reversed(range(1, self.noise_steps)):
            y_target, _ = iter_func(y_target, t)
        return y_target
 
    def conditional_sample(
        self, x, mask, *, x_context, y_context, 
        mask_context, model_fn, num_inner_steps: int=5, 
        method: str="repaint"
    ):
        model_fn.eval()
    
        #somehting is worng with the masking funcitons here. 
        if mask is None:
            mask = torch.zeros_like(x[:, 0])
        if mask_context is None:
            mask_context = torch.zeros_like(x_context[:, 0])
        """
        Note, previously they were using no batch dimension so concatenation was done on dim=0
        but if we use batchees concat needs to be done on dim=1
        """
        #why are we doing this?
        x_augmented = torch.concatenate([x_context, x], dim=1)
        mask_augmented = torch.concatenate([mask_context, mask], dim=1)

        num_context = len(x_context[0]) #is this right? especially at [0]?

        def repaint_inner(yt_target, t):

            yt_context = self.forward(y_context, t)[0] #y* 
            y_augmented = torch.concatenate([yt_context, yt_target], dim=1) #y* union y_t^c
            noise_hat = model_fn.forward(x=x_augmented,y=y_augmented, t=t, mask=mask_augmented)
            y = self.ddpm_backward_step(noise_hat, y_augmented, t)
            # if yt_target.shape[0] != 1:
            #     y = y[:, num_context:, :]
            # else:
            y = y[:, num_context:]

            #one step forward from t-1 to t using repaint
            z = torch.randn_like(y)
            beta_t_minus_one = _extract_into_tensor(self.beta, t - 1, y.shape)
            y = torch.sqrt(1.0 - beta_t_minus_one) * y + torch.sqrt(beta_t_minus_one) * z
            return y
        
        def repaint_outer(y, t):
            for i in range(num_inner_steps):
                val = repaint_inner(y, t)
                assert val.shape == y.shape
                y = val

            #step backward t - > t-1
            yt_context = self.forward(y_context, t)[0]
            y_augmented = torch.concatenate([yt_context, y], dim=1)
            noise_hat = model_fn.forward(x=x_augmented,
                                        y=y_augmented, t=t, mask=mask_augmented)
            
            y = self.ddpm_backward_step(noise_hat, y_augmented, t)
            y = y[:, num_context:]
            return y
        

        #shape required = x.shape[0], x.shape[1], y_context.shape[-1]
        shape_req = (x.shape[0], x.shape[1], y_context.shape[-1])  # Create a tuple for the desired shape
        y_target = torch.randn(shape_req).to(x.device)
        for i in reversed(range(1, self.noise_steps)):
            y_target = repaint_outer(y_target, torch.tensor(i).to(x.device).repeat(shape_req[0]))
        
        y_target = (torch.clamp(y_target, -1, 1) + 1) / 2
        return y_target