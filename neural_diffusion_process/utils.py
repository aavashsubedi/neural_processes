

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