import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0., max_len: int = 5000, concat: bool = False,
                 features_first: bool = False, output_dim=1024, **kwargs):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._dropout = nn.Dropout(p=dropout)
        self._concat = concat
        self._features_first = features_first
        self._max_len = max_len
        self._output_dim = output_dim

        embeddings = torch.zeros(max_len, self._embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self._embedding_dim, 2).float() * (-math.log(10000.0) / self._embedding_dim)
        )
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        embeddings.unsqueeze_(0)

        self.register_buffer("_embeddings", embeddings)

    def _increase_max_len(self, new_max_len: int):
        embeddings = torch.zeros(new_max_len, self._embedding_dim)
        position = torch.arange(0, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self._embedding_dim, 2).float() * (-math.log(10000.0) / self._embedding_dim)
        )
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        embeddings.unsqueeze_(0)

        self._embeddings = embeddings.to(self._embeddings.device)

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        if self._features_first:
            input = input.transpose(1, 2)  # (B, F, T) -> (B, T, F)
        if self._embeddings.shape[1] < input.size(1):
            self._increase_max_len(max(self._embeddings.shape[1] * 2, input.size(1)))
        if self._concat:
            output = torch.cat(
                [input,
                 self._embeddings.type_as(input)[:, :input.size(1), :].repeat(input.size(0), 1, 1)],
                dim=-1
            )
        else:
            output = input + self._embeddings[:, :input.size(1), :]
        if self._features_first:
            output = output.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        return self._dropout(output)

    def tick(self,
             embedding: Tensor,
             position: Tensor):
        assert not self._features_first
        batch_size = embedding.size(0)
        div_term = torch.exp(
            torch.arange(0, self._embedding_dim, 1,
                         device=embedding.device).float() * (-math.log(10000.0) / self._embedding_dim)
        ).expand(batch_size, -1)
        bias = torch.arange(0, 2, 1, device=embedding.device).float() * math.pi / 2
        bias = bias.expand(self._embedding_dim // 2, -1).reshape(self._embedding_dim)

        encoding = torch.sin((position.unsqueeze(-1) + 1) * div_term + bias)
        if self._concat:
            return torch.cat([embedding, encoding], dim=-1)
        else:
            return embedding + encoding
