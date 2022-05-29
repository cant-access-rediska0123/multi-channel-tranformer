from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Coalescence(nn.Module):
    def __init__(self, window_size: int, stride: int, input_dim: int, **kwargs):
        super().__init__()
        self._window_size = window_size
        self._stride = stride
        self._output_dim = window_size * input_dim

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: Tensor, input_lengths: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        input = input.transpose(1, 2).contiguous()  # (B, F, T) -> (B, T, F)
        batch_size, max_seq_len, input_dim = input.size()

        output = []
        for i in range(0, max_seq_len, self._stride):
            j = min(i + self._window_size, max_seq_len)
            if j - i < self._window_size:
                window = torch.zeros(batch_size, self._window_size, input_dim).type_as(input)
                window[:, :j - i, :] = input[:, i:j, :]
            else:
                window = input[:, i:j, :]
            output.append(window.view(batch_size, 1, -1))
        output = torch.cat(output, 1)
        output = output.transpose(1, 2)  # (B, T, F) -> (B, F, T)

        if input_lengths is not None:
            input_lengths = ((input_lengths + self._stride - 1) // self._stride).long()

        return output, input_lengths
