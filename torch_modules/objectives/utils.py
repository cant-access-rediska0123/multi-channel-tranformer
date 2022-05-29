import torch


def reduce_loss(loss, reduction: str, drop_nans: bool):
    if drop_nans:
        finite_mask = torch.isfinite(loss)
        finite_ratio = finite_mask.float().mean()
        if finite_ratio > 0.75:
            loss = loss[finite_mask]
            if reduction == 'mean':
                loss *= finite_ratio

    if reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'mean':
        return torch.mean(loss)
    raise ValueError('Unknown reduction type: {}'.format(reduction))
