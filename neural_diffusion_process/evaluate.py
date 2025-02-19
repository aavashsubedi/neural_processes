import torch
import matplotlib.pyplot as plt
import os
from torch.func import vmap
import wandb
from dataloader import get_data, get_context_mask


def evaluate(cfg, dataloader, diffusion, ema_model,
            results_dir, epoch, device="cuda", percentage=0.6, 
            generate_plots=True):

    loss_array = []
    get_idx_keep = lambda x: torch.where(x == 33,
     torch.ones_like(x, dtype=torch.bool),
      torch.zeros_like(x, dtype=torch.bool))

    def func(x_grid, y_grid, image_size=cfg.img_size, p=cfg.percentage):
            context_mask = get_context_mask(image_size=image_size, p=p)
            x_grid = x_grid.unsqueeze(0)
            y_grid = y_grid.unsqueeze(0)
            x_context = x_grid[:, get_idx_keep(context_mask)]
            y_context = y_grid[:, get_idx_keep(context_mask)]
            mask_context = torch.zeros_like(x_grid[:, get_idx_keep(context_mask)][..., 0])
            context_only = get_idx_keep(context_mask).unsqueeze(0).float()
            return x_context, y_context, mask_context, context_only
    vmap_func = vmap(func, randomness="same")

    for (x_grid, y_val, _) in dataloader:
        if len(loss_array) > 2:
            continue
        x_grid, y_val = x_grid.to(device), y_val.to(device)

        with torch.no_grad():
            #take first 5 datapoints in the batch
            x_grid, y_val = x_grid[:50], y_val[:50]
            x_context, y_context, mask_context, context_only = vmap_func(x_grid, y_val)
            x_context, y_context = x_context.squeeze(1), y_context.squeeze(1)
            mask_context, context_only = mask_context.squeeze(1), context_only.squeeze(1)
            samples = diffusion.conditional_sample(model_fn=ema_model,
                                            x = x_grid, x_context = x_context,
                                            y_context = y_context,
                                            mask_context = mask_context,
                                            mask=None
                                            )
            #is this even a good evaluation? If we iteratively go back then we might not get the same results
            loss = (samples - y_val).pow(2).squeeze(-1)
            loss = loss * (1 - context_only.to(device))
            loss = loss.sum() / (1 - context_only.to(device)).sum()
            loss_array.append(loss.item())
            wandb.log({"eval_loss": loss.item()})
        
    loss = torch.tensor(loss_array).mean()
    wandb.log({"MSE_eval": loss.item()})

    if not generate_plots:
        return loss_array

    with torch.no_grad():
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

def evaluate_flow(cfg, dataloader, diffusion, ema_model, results_dir, epoch,
                    device="cuda", percentage=0.6, generate_plots=True):
    
    loss_array = []
    get_idx_keep = lambda x: torch.where(x == 33,
        torch.ones_like(x, dtype=torch.bool),
        torch.zeros_like(x, dtype=torch.bool))

    def func(x_grid, y_grid, image_size=cfg.img_size, p=cfg.percentage):
            context_mask = get_context_mask(image_size=image_size, p=p)
            x_grid = x_grid.unsqueeze(0)
            y_grid = y_grid.unsqueeze(0)
            x_context = x_grid[:, get_idx_keep(context_mask)]
            y_context = y_grid[:, get_idx_keep(context_mask)]
            mask_context = torch.zeros_like(x_grid[:, get_idx_keep(context_mask)][..., 0])
            context_only = get_idx_keep(context_mask).unsqueeze(0).float()
            return x_context, y_context, mask_context, context_only

    vmap_func = vmap(func, randomness="same")

    for (x_grid, y_val, _) in dataloader:
        if len(loss_array) > 2:
            continue
        x_grid, y_val = x_grid.to(device), y_val.to(device)

        with torch.no_grad():
            #take first 5 datapoints in the batch
            x_grid, y_val = x_grid[:3], y_val[:3]
            #y_context ranges from [0, 1]
            x_context, y_context, mask_context, context_only = vmap_func(x_grid, y_val)
            x_context, y_context = x_context.squeeze(1), y_context.squeeze(1)
            mask_context, context_only = mask_context.squeeze(1), context_only.squeeze(1)
            samples = diffusion.conditional_sample(model_fn=ema_model,
                                            x = x_grid, x_context = x_context,
                                            y_context = y_context,
                                            mask_context = mask_context, mask=None
                                            )
            subset_samples = samples[:, 0:len(y_context[0]), 0]
            loss = (samples - y_val).pow(2).squeeze(-1)
            loss = loss * (1 - context_only.to(device))
            loss = loss.sum() / (1 - context_only.to(device)).sum()
            loss_array.append(loss.item())
            wandb.log({"eval_loss": loss.item()})
        
    loss = torch.tensor(loss_array).mean()
    wandb.log({"MSE_eval": loss.item()})


    if not generate_plots:
        return loss_array

    with torch.no_grad():
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
