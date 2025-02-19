import torch 
import torch.nn as nn
from diffusion_base import Diffusion_Base
from utils import _extract_into_tensor


class NFP(nn.Module):

    def __init__(self, cfg, img_size=64, device="cuda"):
        super(NFP, self).__init__()
        self.cfg = cfg
        self.img_size = img_size
        self.device = device
        self.noise_steps = cfg.noise_steps
    
    def sample_timesteps(self, n):
        return torch.rand(size=(n,))

    def forward(self, x, t):
        """
        Takes the image x to the noise level t
        """
        x_1 = x 
        x_0 = torch.randn_like(x)
        t = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t) * x_0 + t * x_1 #noise output
        return x_t, x_0 #prev doing d_x was just feeding it noise       
    
    def backward_step(self, velocity_estimate, yt, t_start, t_end):
        t_start, t_end = t_start.unsqueeze(-1).unsqueeze(-1), t_end.unsqueeze(-1).unsqueeze(-1) 
        """Main thing i changed was the direction t_end - t_start"""
        
        yt_minus_one = yt + (t_end - t_start) * velocity_estimate #velocity_estimate = fn(x_t, t_start)
        return yt_minus_one
    
    def sample(self, x, mask, model, output_dim=1): #what is x here?
        y_target = torch.randn(len(x), output_dim)
        if mask is None:
            mask = torch.zeros_like(x[:, 0])
        def iter_func(y, t_start, t_end):
            predicted_velocity = model(x, y, t_old, mask=mask)
            y = self.backward_step(predicted_velocity, y, t_start, t_end)
            return y
        t = torch.linspace(0, self.noise_steps, self.noise_steps + 1)
        t = t/t.max() #t_n = n/N. t_max = 1, t_min = n/N 
        """Issue here"""
        for i in reversed(range(0, self.noise_steps)): #ends at 1 because otherwise t_new would be -ve
            #this goes from t_max to 0. Checked on colab. End points are not included!
            t_start = t[i + 1]
            t_end = t[i]
            y_target = iter_func(y_target, t_start=t_start, t_end=t_end)
        return y_target 
    
    def conditional_sample(
        self, x, mask, *, x_context, y_context, 
        mask_context, model_fn, num_inner_steps: int=5, 
        method: str="repaint", debug_mode=False,
    ):
        model_fn.eval()
        #somehting is worng with the masking funcitons here. 
        if mask is None:
            if debug_mode:
                mask = torch.zeros_like(x[:, :, 0])
            else:
                mask = torch.zeros_like(x[:, 0])
        if mask_context is None:
            mask_context = torch.zeros_like(x_context[:, 0])
        """
        Note, previously they were using no batch dimension
        so concatenation was done on dim=0
        but if we use batchees concat needs to be done on dim=1
        """
        #why are we doing this?

        
        x_augmented = torch.concatenate([x_context, x], dim=1) #check the shape of this or what its supposed to be?
        mask_augmented = torch.concatenate([mask_context, mask], dim=1)

        num_context = len(x_context[0]) #is this right? especially at [0]? -
        """
         assumes we have the same context size for all images
        """
        def repaint_inner(yt_target, t_start, t_end):
            """Feels wrong."""
            yt_context = self.forward(y_context, t_end)[0] # check where if we should forward to t_{n+1}
            y_augmented = torch.concatenate([yt_context, yt_target], dim=1) #y* union y_t^c
           
            velocity_estimate = model_fn.forward(x=x_augmented, y=y_augmented, 
                                        t=t_start, mask=mask_augmented) 

            y = self.backward_step(velocity_estimate, y_augmented, t_start=t_start, t_end=t_end)
            y = y[:, num_context:] #select only the target part
            #one step forward from t-1 to t using repaint
            """
            Change here for finding t-1 using t
            
            This will be something like, 
            y = y + (t_n - t{n+1}) * velocity_estimate
            """
            y = y + (t_end - t_start).unsqueeze(-1).unsqueeze(-1) * velocity_estimate[:, num_context:]            
            return y
        def repaint_outer(y, t_start, t_end):
            for i in range(num_inner_steps):
                val = repaint_inner(y, t_start=t_start, t_end=t_end)
                assert val.shape == y.shape
                y = val

            #step backward t - > t-1
            """Change here."""
            yt_context = self.forward(y_context, t_end)[0] # check where if we should forward to t_{n+1}
            y_augmented = torch.concatenate([yt_context, y], dim=1)

            ###Here the prediction should be #z_t = z_t{n+1} + (t_n - t_{n+1}) * model_fn(z_t{n+1}, t_n) (obviously give the augmented & masked)
            # noise_hat = model_fn.forward(x=x_augmented,
            #                             y=y_augmented, t=t, mask=mask_augmented)
            
            velocity_estimate = model_fn.forward(x=x_augmented, y=y_augmented,
                                                 t=t_start, mask=mask_augmented)
            """"""
            y = self.backward_step(velocity_estimate, y_augmented,
                                     t_start=t_start, t_end=t_end)
            y = y[:, num_context:]
            return y

        #shape required = x.shape[0], x.shape[1], y_context.shape[-1]
        shape_req = (x.shape[0], x.shape[1], y_context.shape[-1])  # Create a tuple for the desired shape
        y_target = torch.randn(shape_req).to(x.device)
        
        t = torch.linspace(0, self.noise_steps, self.noise_steps + 1).to(x.device)
        t = t/t.max() #t_n = n/N. t_max = 1, t_min = n/N 

        for i in reversed(range(0, self.noise_steps)):
            # t_old = t[i + 1]
            # t_new = t[i]
            y_target = repaint_outer(y_target, t_start=t[i + 1].repeat(shape_req[0]),
                     t_end=t[i].repeat(shape_req[0]))
        
        
        y_target = (torch.clamp(y_target, -1, 1) + 1) / 2 #do we need to do this?
        return y_target