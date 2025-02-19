import torch
import torch.nn as nn
from dataloader import get_data, get_context_mask
from tqdm import tqdm
from attention import AttentionModel
from NFP import NFP
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
from evaluate import evaluate_flow

KEEP = 33  # random number
NOT_KEEP = 44  # random number

def setup_dir(cfg, run_name):
    model_dir = os.path.join(cfg.working_dir, "models",
     run_name)
    os.makedirs(model_dir, exist_ok=True)
    results_dir = os.path.join(cfg.working_dir,"results",
     run_name)
    os.makedirs(results_dir, exist_ok=True)
    return model_dir, results_dir

def train_flow(cfg, dataset_path, run_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_data(image_size=cfg.img_size, path=cfg.data_dir, batch_size=cfg.batch_size)
    val_dataloader = get_data(image_size=cfg.img_size, path=cfg.val_dir, batch_size=cfg.batch_size*1)
    
    model = AttentionModel(cfg).to(device).to(torch.float32)
    diffusion = NFP(cfg=cfg, img_size=cfg.img_size, device=device)
       
    mse = nn.MSELoss(reduction="none").to(device)
    mae = nn.L1Loss(reduction="none").to(device)
    if cfg.schedule == "cosine":
        optimizer = torch.optim.Adam(model.parameters(), lr=1) #initial will get multiplied by scheduler
        scheduler = warmup_cosine_decay_schedule(
            optimizer, cfg.init_lr, cfg.peak_lr, cfg.end_lr,
            cfg.num_warmup_epochs, cfg.num_decay_epochs,
            len(dataloader)
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = None

    model_dir, results_dir = setup_dir(cfg, run_name)
    
    ema = EMA(beta=cfg.ema_rate)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    get_idx_keep = lambda x: torch.where(x == 33,
     torch.ones_like(x, dtype=torch.bool),
      torch.zeros_like(x, dtype=torch.bool))

 
    # evaluate_flow(cfg, val_dataloader, diffusion, ema_model, results_dir,
    #           epoch=0, device=device, generate_plots=True)
            
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dataloader)
        for i, (x_grid, y_val, _)  in enumerate(pbar):
            x_grid, y_val = x_grid.to(device), y_val.to(device)
            
            t = diffusion.sample_timesteps(x_grid.shape[0]).to(device)
            target, noise = diffusion.forward(y_val, t) #target is t
            dx_t = y_val - noise
            #for training mask = None
            predicted_velocity = model(x_grid, target, t, mask=None) #equivalent to flow(x_t, t, target/context etc..)
            """
            #loss = || X_1 - X_0 - v(x_t, t) ||^2
            #loss = || Noise - y_val - v(x_t, t) ||^2
            loss = loss_fn(flow(x_t, t), d)
            """
            loss = mse(predicted_velocity, dx_t)
            loss = loss.mean(dim=[-1, -2]) #leave N dimension
            loss = loss.mean()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            ema.step_ema(model, ema_model)

            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")
            pbar.set_postfix(MSE=loss.item())
            wandb.log({"MSE": loss.item()})
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

        if (epoch + 1) % cfg.save_every == 0:
            evaluate_flow(cfg, val_dataloader, diffusion, ema_model, results_dir,
              epoch=epoch, device=device, generate_plots=True)
            torch.save(model.state_dict(), os.path.join(model_dir,
                 f"ckpt_{epoch}.pt"))
    return 0