import torch
import torch.nn as nn
from dataloader import get_data, get_context_mask
from tqdm import tqdm
from attention import AttentionModel
from neural_diffusion import NDP
from diffusion_base import EMA
import wandb
import os
import copy
from sampling import sample_n_conditionals
from scheduler import warmup_cosine_decay_schedule
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from torch.func import vmap

KEEP = 33  # random number
NOT_KEEP = 44  # random number

def train_mnist(cfg, dataset_path, run_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_data(image_size=cfg.img_size, path=cfg.data_dir, batch_size=cfg.batch_size)
    model = AttentionModel(cfg).to(device).to(torch.float32)
    diffusion = NDP(cfg=cfg, img_size=cfg.img_size, device=device,
     bw=cfg.bw, use_cosine=cfg.use_cosine)
    mse = nn.MSELoss(reduction="none").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1) #initial will get multiplied by scheduler

    model_dir = os.path.join(cfg.working_dir, "models",
     run_name)
    os.makedirs(model_dir, exist_ok=True)
    results_dir = os.path.join(cfg.working_dir,"results",
     run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    ema = EMA(beta=cfg.ema_rate)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    get_idx_keep = lambda x: torch.where(x == 33,
     torch.ones_like(x, dtype=torch.bool),
      torch.zeros_like(x, dtype=torch.bool))

    scheduler = warmup_cosine_decay_schedule(
        optimizer, cfg.init_lr, cfg.peak_lr, cfg.end_lr,
        cfg.num_warmup_epochs, cfg.num_decay_epochs,
        len(dataloader)
    )
    
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dataloader)

        for i, (x_grid, y_val, _)  in enumerate(pbar):
            x_grid, y_val = x_grid.to(device), y_val.to(device) 
            
            t = diffusion.sample_timesteps(x_grid.shape[0]).to(device)
            optimizer.zero_grad()     
            #x_grid: [b, h*w, 2], y_val: [b, h*w, 1]
        
            #for training mask = None
            yt, noise = diffusion.forward(y_val, t)
            predicted_noise = model(x_grid, yt, t, mask=None)
            loss = mse(predicted_noise, noise)
            loss = loss.mean(dim=[-1, -2]) #leave N dimension
            loss = loss.mean()
            
            loss.backward()
       
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.step_ema(model, ema_model)

            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")
            pbar.set_postfix(MSE=loss.item())
            wandb.log({"MSE": loss.item()})
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

        if (epoch + 1) % 1 == 0:
            # x_grid = x_grid[ ...].unsqueeze(0)
            # y_val = y_val[0, ...].unsqueeze(0)

            #vmap over first dimension
            
            def func(x_grid, y_grid, image_size=cfg.img_size, p=0.8):
                context_mask = get_context_mask(image_size=image_size, p=p)
                x_grid = x_grid.unsqueeze(0)
                y_grid = y_grid.unsqueeze(0)
                x_context = x_grid[:, get_idx_keep(context_mask)]
                y_context = y_grid[:, get_idx_keep(context_mask)]
                mask_context = torch.zeros_like(x_grid[:, get_idx_keep(context_mask)][..., 0])
                context_only = get_idx_keep(context_mask).unsqueeze(0).float()
                return x_context, y_context, mask_context, context_only
                # samples = diffusion.conditional_sample(model_fn=ema_model,
                #                             x = x_grid, x_context = x_context,
                #                             y_context = y_context,
                #                             mask_context = mask_context, mask=None
                #                             )
                # return samples
            vmap_func = vmap(func, randomness="same")
            model.eval()
            #take first 5 datapoints in the batch
            x_grid, y_val = x_grid[:5], y_val[:5]
            x_context, y_context, mask_context, context_only = vmap_func(x_grid, y_val)
            x_context, y_context = x_context.squeeze(1), y_context.squeeze(1)
            mask_context, context_only = mask_context.squeeze(1), context_only.squeeze(1)
            samples = diffusion.conditional_sample(model_fn=ema_model,
                                            x = x_grid, x_context = x_context,
                                            y_context = y_context,
                                            mask_context = mask_context, mask=None
                                            )
            loss = (samples - y_val).pow(2).squeeze(-1)
            loss = loss * (1 - context_only.to(device))
            loss = loss.sum() / (1 - context_only.to(device)).sum()
            wandb.log({"MSE_eval": loss.item()})
            model.train()

           
        if (epoch + 1) % cfg.save_every == 0:
            # if True:
            #     import pdb; pdb.set_trace()
            #     context_mask = get_context_mask(image_size=cfg.img_size, p=0.7)
            #     x_context = x_grid[:, get_idx_keep(context_mask)]
            #     y_context = y_val[:, get_idx_keep(context_mask)]
            #     mask_context = torch.zeros_like(x_target[:, get_idx_keep(context_mask)][..., 0])
            #     samples = sample_n_conditionals(
            #         diffusion_process=diffusion,
            #         model=ema_model,
            #         x = x_grid,
            #         x_context = x_context,
            #         y_context = y_context,
            #     )
            if True:
                #for plotting:
                model.eval()
                x_vals = x_grid[0].cpu().numpy() #784, 2
                y_vals = y_val[0].squeeze(-1).reshape(28, 28).cpu().numpy() #784, 1
                context_mask_vals = context_only[0].cpu().numpy()

                context_x_vals = x_vals[context_mask_vals == 1]
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                # Plot 1: Heatmap using imshow
                ax = axes[0]
                im = ax.imshow(y_vals, extent=[-2, 2, -2, 2], cmap="binary_r", )
                fig.colorbar(im, ax=ax, label="Y-Values")
                ax.set_title("Heatmap (imshow)")
             
                grid_size = (28, 28)  # Height, Width
                h, w = grid_size
                extent = [-2, 2, -2, 2]
                pixel_width = (extent[1] - extent[0]) / w  # Width of a pixel
                pixel_height = (extent[3] - extent[2]) / h  # Height of a pixel

                # Convert pixel size to scatter marker size in points^2
                marker_size = ((72.0 * pixel_width / fig.get_size_inches()[0]) ** 2)

                # Plot 2: Scatter plot of context points
                ax = axes[1]
                scatter = ax.scatter(
                    context_x_vals[:, 0], context_x_vals[:, 1], 
                    c=y_vals.flatten()[context_mask_vals==1], s=marker_size*300, marker="s",
                )
                #flip the y-axis
                fig.colorbar(scatter, ax=ax, label="Y-Values")
                ax.legend()       
                ax.set_xlim(-2, 2)
                ax.set_ylim(2, -2)

                #Plot 3:Scatter plot of the labels
                ax = axes[2]
                samples = samples[0].squeeze(-1).reshape(28, 28).cpu().numpy()
                ax.imshow(samples, extent=[-2, 2, -2, 2], cmap="binary_r")
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"plot_{epoch}.png"), dpi=300)
                plt.show()
                fig.clf()
                model.train()
            torch.save(model.state_dict(), os.path.join(model_dir,
                 f"ckpt_{epoch}.pt"))
    return 0