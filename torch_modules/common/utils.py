import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_attention_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def get_padding_mask(lengths, max_length=None):
    if max_length is None:
        max_length = torch.max(lengths).item()
    batch_size = lengths.size(0)
    padding_mask = torch.arange(
        max_length
    ).type_as(  # move to the right device
        lengths
    ).view(  # reshape to (1, T)-shaped tensor
        1, max_length
    ).expand(
        batch_size, -1
    ) >= lengths.view(  # expand to (B, T)-shaped tensor
        batch_size, 1
    ).expand(
        -1, max_length
    )
    return padding_mask


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")
