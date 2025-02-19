import hydra
import omegaconf
import os
from train_mnist import train_mnist
from train_flow import train_flow
from utils import setup_wandb
import torch
import torch.nn as nn
from dataloader import get_data, get_context_mask
from tqdm import tqdm
from attention import AttentionModel
from NFP import NFP
from diffusion_base import EMA
import wandb
import copy
from sampling import sample_n_conditionals
from scheduler import warmup_cosine_decay_schedule
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from torch.func import vmap
from evaluate import evaluate_flow


def check_overlap(array1, array2):
    #check if there any values in array1 that are in array2
    return bool(set(array1).intersection(array2))
def debug_flow_eval(cfg, dataloader, diffusion, ema_model, results_dir=0, epoch=0,
                    device="cuda", percentage=0.6, generate_plots=True):
    
    loss_array = []
    context_loss_array = []
    get_idx_keep = lambda x: torch.where(x == 33,
        torch.ones_like(x, dtype=torch.bool),
        torch.zeros_like(x, dtype=torch.bool))

    def func(x_grid, y_grid, image_size=cfg.img_size, p=cfg.percentage):
            context_mask = get_context_mask(image_size=image_size, p=p)
            x_grid = x_grid.unsqueeze(0)
            y_grid = y_grid.unsqueeze(0)
            x_context = x_grid[:, get_idx_keep(context_mask)]
            x_pred = x_grid[:, ~get_idx_keep(context_mask)]
            y_context = y_grid[:, get_idx_keep(context_mask)]
            y_pred = y_grid[:, ~get_idx_keep(context_mask)]
            mask_context = torch.zeros_like(x_grid[:, get_idx_keep(context_mask)][..., 0])
            context_only = get_idx_keep(context_mask).unsqueeze(0).float()
            return x_context, y_context, mask_context, context_only, x_pred, y_pred

    vmap_func = vmap(func, randomness="same")

    for (x_grid, y_val, _) in dataloader:
        if len(loss_array) > 2:
            continue
        x_grid, y_val = x_grid.to(device), y_val.to(device)

        with torch.no_grad():
            #take first 5 datapoints in the batch
            x_grid, y_val = x_grid[:2], y_val[:2]
            #y_context ranges from [0, 1]
            x_context, y_context, mask_context, context_only, x_pred, y_pred = vmap_func(x_grid, y_val)
            x_context, y_context, x_pred = x_context.squeeze(1), y_context.squeeze(1), x_pred.squeeze(1)
            
            mask_context, context_only = mask_context.squeeze(1), context_only.squeeze(1)
            mask = torch.zeros_like(x_pred[:, :, 0])
            samples = diffusion.conditional_sample(model_fn=ema_model,
                                            x = x_pred, x_context = x_context,
                                            y_context = y_context,
                                            mask_context = mask_context, mask=mask,
                                            )
            #here the mean converges to 1 and the standard deviation is 0. so it just gets clamped down.
            subset_samples = samples[:, 0:len(y_context[0]), 0]
            loss = (samples - y_pred.squeeze(1)).pow(2).squeeze(-1).mean()
            loss_array.append(loss.item())
    loss = torch.tensor(loss_array).mean()
    print(f"Loss: {loss}")
    results_dir = "/home/users/asubedi/neural_processes/neural_processes/neural_diffusion_process/working_dir/debug"
    with torch.no_grad():
        x_vals = torch.cat([x_context, x_pred], dim=1)[0].cpu().numpy() #784, 2
        import pdb; pdb.set_trace()
        y_vals = torch.cat([y_context,samples], dim=1)[0].squeeze(-1).reshape(28, 28).cpu().numpy() #784, 1
        #y_vals = y_val[0].squeeze(-1).reshape(28, 28).cpu().numpy() #784, 1
        context_mask_vals = context_only[0].cpu().numpy()
        context_mask_vals = torch.cat([torch.ones_like(y_context[0]), torch.zeros_like(samples[0])], dim=0).squeeze(-1).cpu().numpy()
        context_x_vals = x_vals[context_mask_vals == 1]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Plot 1: Heatmap using imshow
        ax = axes[0]
        im = ax.imshow(y_val[0].reshape(28, 28).squeeze(-1).cpu().numpy(), extent=[-2, 2, -2, 2], cmap="binary_r", )
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
        samples = y_vals
        ax.imshow(samples, extent=[-2, 2, -2, 2], cmap="binary_r")
        #set a colorbar
        fig.colorbar(im, ax=ax, label="Y-Values")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"plot_{epoch}.png"), dpi=300)
        plt.show()
        fig.clf()
@hydra.main(version_base='1.3', config_path="",
             config_name="config.yaml")
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_data(image_size=cfg.img_size, path=cfg.data_dir, batch_size=cfg.batch_size)
    val_dataloader = get_data(image_size=cfg.img_size, path=cfg.val_dir, batch_size=cfg.batch_size*1)
    
    model = AttentionModel(cfg).to(device).to(torch.float32)
    diffusion = NFP(cfg=cfg, img_size=cfg.img_size, device=device)
    model.load_state_dict(torch.load("/home/users/asubedi/neural_processes/neural_processes/neural_diffusion_process/working_dir/models/nfp_3/ckpt_119.pt"))
    debug_flow_eval(cfg, val_dataloader, diffusion, model)


if __name__ == "__main__":
    main()