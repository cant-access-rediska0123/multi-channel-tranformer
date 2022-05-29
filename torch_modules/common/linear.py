import torch.nn as nn


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def input_dim(self) -> int:
        return self.in_features

    def output_dim(self) -> int:
        return self.out_features
