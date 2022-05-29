import torch.nn as nn

from torch_modules.objectives.utils import reduce_loss


class CrossEntropyLoss:
    def __init__(self, pad_id=0, reduction='sum', indices_weight=None, drop_nans=False):
        if indices_weight is not None:
            self._criterion = nn.CrossEntropyLoss(weight=indices_weight, reduction='none')
        else:
            self._criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
        self._reduction = reduction
        self._drop_nans = drop_nans

    def __call__(self, logits, targets):
        pred_flat = logits.reshape((-1, logits.shape[-1]))  # (B * T) x vocab_size
        target_flat = targets.reshape((-1,))  # (B * T)
        loss = self._criterion(pred_flat, target_flat)

        loss = reduce_loss(loss, self._reduction, self._drop_nans)
        return loss
