"""Model weights and resumable training checkpoints."""
import torch


def save_weights(model, path):
    torch.save(model.state_dict(), path)


def load_weights(model, path, device=None):
    model.load_state_dict(torch.load(path, map_location=device))


def save_checkpoint(path, *, epoch, model, optimizer, scheduler,
                    train_losses, val_losses, param_history, loss_state):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'train_loss_dict': train_losses,
        'val_loss_dict': val_losses,
        'param_history': param_history,
        'loss_state': loss_state,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler=None, device='cuda'):
    """Restore model, optimizer and scheduler in place. Returns (resume epoch, checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    resume_epoch = checkpoint['epoch'] + 1
    if scheduler is not None and checkpoint.get('scheduler_state'):
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scheduler.last_epoch = resume_epoch
    return resume_epoch, checkpoint