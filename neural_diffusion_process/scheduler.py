import math
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine_decay_schedule(
    optimizer, init_lr, peak_lr, end_lr,
     warmup_epochs, decay_epochs, steps_per_epoch
):
    """
    Create a learning rate schedule based on warmup epochs and decay epochs.

    Args:
        optimizer: PyTorch optimizer.
        init_lr: Initial learning rate at the start of warmup.
        peak_lr: Peak learning rate after warmup.
        end_lr: Final learning rate after decay.
        warmup_epochs: Number of warmup epochs.
        decay_epochs: Number of decay epochs.
        steps_per_epoch: Number of steps in each epoch.

    Returns:
        LambdaLR scheduler.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = decay_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup phase: Linearly increase the learning rate
            return init_lr + (peak_lr - init_lr) * (current_step / warmup_steps)
        elif current_step < warmup_steps + decay_steps:
            # Decay phase: Apply cosine decay
            decay_step = current_step - warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))
            return end_lr + (peak_lr - end_lr) * cosine_decay
        else:
            # After decay: Learning rate stays at the end value
            return end_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)