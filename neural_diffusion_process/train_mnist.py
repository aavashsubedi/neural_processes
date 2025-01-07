import torch
import torch.nn as nn
from dataloader import get_data
from tqdm import tqdm
from attention import AttentionModel
from neural_diffusion import NDP
import wandb
def train_mnist(cfg, dataset_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_data(image_size=cfg.img_size, path=cfg.data_dir)
    model = AttentionModel(cfg).to(device).to(torch.float32)
    diffusion  = NDP(cfg=cfg, img_size=cfg.img_size, device=device,
     bw=cfg.bw, use_cosine=cfg.use_cosine)
    mse = nn.MSELoss(reduction="none").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dataloader)

        for i, (x_grid, y_val)  in enumerate(pbar):
            # images, labels = images.to(device), labels.to(device)
            x_grid, y_val = x_grid.to(device), y_val.to(device)
            t = diffusion.sample_timesteps(x_grid.shape[0]).to(device)
            
            # model(x_grid, y_val, t)
            #x_grid: [b, h*w, 2], y_val: [b, h*w, 1]
            
            mask = torch.zeros_like(t) # we will figure out mask later
            yt, noise = diffusion.forward(y_val, t)
            predicted_noise = model(x_grid, yt, t, mask=mask)
            loss = mse(predicted_noise, noise)
            loss = loss.mean(dim=[-1, -2]) #leave N dimension
            loss = loss * (1.0 - mask)
            # non_zero_mask = torch.count_non_zero(mask)
            loss = loss.mean()# / non_zero_mask
            #sum mse leaving the first dimension
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")
            pbar.set_postfix(MSE=loss.item())
            wandb.log({"MSE": loss.item()})


    return 0