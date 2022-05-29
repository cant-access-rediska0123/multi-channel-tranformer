import torch
import torch.nn as nn

from torch_modules.objectives.utils import reduce_loss


class CTCLoss:
    def __init__(self,
                 blank_id,
                 reduction='mean',
                 drop_nans=False,
                 **kwargs):
        self._criterion = nn.CTCLoss(blank=blank_id,
                                     reduction=reduction)
        self._reduction = reduction
        self._drop_nans = drop_nans

    def __call__(self,
                 log_probs: torch.Tensor,
                 targets: torch.Tensor,
                 target_length: torch.Tensor):
        input_length = torch.full(size=(log_probs.shape[0],), fill_value=log_probs.shape[1], dtype=torch.int32)
        target_length = target_length.int()
        targets = targets.int()
        loss = self._criterion(log_probs.transpose(1, 0), targets,
                               input_length, target_length)
        loss = reduce_loss(loss, self._reduction, self._drop_nans)
        return loss
