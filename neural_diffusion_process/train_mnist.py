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

KEEP = 33  # random number
NOT_KEEP = 44  # random number

def train_mnist(cfg, dataset_path, run_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_data(image_size=cfg.img_size, path=cfg.data_dir)
    model = AttentionModel(cfg).to(device).to(torch.float32)
    diffusion = NDP(cfg=cfg, img_size=cfg.img_size, device=device,
     bw=cfg.bw, use_cosine=cfg.use_cosine)
    mse = nn.MSELoss(reduction="none").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

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

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dataloader)

        for i, (x_grid, y_val)  in enumerate(pbar):
        
            # mask = get_context_mask(image_size=cfg.img_size, p=0.7) #[N,]
            # print(mask.shape)

            # image = diffusion.sample(x_grid, mask=None, model=ema_model)
            
            optimizer.zero_grad()

            # images, labels = images.to(device), labels.to(device)
            x_grid, y_val = x_grid.to(device), y_val.to(device)
            t = diffusion.sample_timesteps(x_grid.shape[0]).to(device)
            
            # model(x_grid, y_val, t)
            #x_grid: [b, h*w, 2], y_val: [b, h*w, 1]
        
            #for training mask = None
            yt, noise = diffusion.forward(y_val, t)
            predicted_noise = model(x_grid, yt, t, mask=None)
            loss = mse(predicted_noise, noise)
            loss = loss.mean(dim=[-1, -2]) #leave N dimension
            loss = loss * (1.0 - mask)
            loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            ema.step_ema(model, ema_model)

            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")
            pbar.set_postfix(MSE=loss.item())
            wandb.log({"MSE": loss.item()})

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

                torch.save(model.state_dict(), os.path.join(model_dir,
                 f"ckpt_{epoch}.pt"))
    return 0