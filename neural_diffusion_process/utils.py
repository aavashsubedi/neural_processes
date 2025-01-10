

import wandb
import omegaconf

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """

    res = arr.to(device=timesteps.device)[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res
def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'project': cfg.project_name, 'config': config_dict, 'reinit': True, 'mode': cfg.wandb,
              'settings': wandb.Settings(_disable_stats=True)}
    run = wandb.init(**kwargs)
    #wandb.save('*.txt')
    #run.save()
    return cfg, run

def plotting_function(x_grid, x_context, y_context, y_target):
    """
    x_grid: [b, h*w, 2]
    x_context: [b, context h*w, 2]
    y_context: [b, context h*w, 1]
    y_target: [b, h*w, 1]
    """
    x_grid = x_grid[0, ...].cpu().numpy()
    x_context = x_context[0, ...].cpu().numpy()
    y_context = y_context[0, ...].cpu().numpy()
    y_target = y_target[0, ...].cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(y_image, extent=[-2, 2, -2, 2], origin="lower", cmap='viridis')
    plt.colorbar(label="Y-Values")
    plt.scatter(x_context[:, 0], x_context[:, 1], c=y_context[:, 0], cmap='viridis', edgecolor='white', s=50, label="Context Points")
    plt.title("Y-Values on Grid with Context Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()