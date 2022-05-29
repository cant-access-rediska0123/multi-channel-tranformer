import torch.nn as nn
from torch import Tensor


class Conv(nn.Module):
    def __init__(self, stride: int,
                 kernel: int,
                 in_channels: int,
                 out_channels: int, **kwargs):
        super().__init__()

        self._out_channels = out_channels
        self.conv = nn.Conv1d(stride=(stride,), kernel_size=(kernel,),
                              in_channels=in_channels, out_channels=out_channels)
        self.relu = nn.ReLU()

    def output_dim(self) -> int:
        return self._out_channels

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        # (B, F, T)
        out = self.conv(input)
        return self.relu(out)
