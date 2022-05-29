from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_modules.common.utils import get_activation_fn, get_attention_mask, get_clones, get_padding_mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.dropout2(src)
        src = residual + src
        if not self.normalize_before:
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, model_definition: dict, **kwargs):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=model_definition["d_model"],
                                                num_heads=model_definition["num_heads"],
                                                dim_feedforward=model_definition["dim_feedforward"],
                                                dropout=model_definition["dropout"],
                                                activation=model_definition["activation"],
                                                normalize_before=model_definition["normalize_before"])
        self.num_layers = model_definition["num_layers"]
        self.d_model = model_definition["d_model"]
        self.layers = get_clones(encoder_layer, self.num_layers)
        self.final_norm = None
        if model_definition["final_norm"]:
            self.final_norm = nn.LayerNorm(self.d_model)
        self.use_triangular_mask = model_definition.get("use_triangular_mask", False)
        self.triangular_mask = None

    def input_dim(self) -> int:
        return self.d_model

    def output_dim(self) -> int:
        return self.d_model

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor,
                final_norm: bool = True, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        output = input.transpose(1, 2).transpose(0, 1)  # (B, F, T) -> (B, T, F) -> (T, B, F)
        padding_mask = None
        if input_lengths is not None:
            padding_mask = get_padding_mask(input_lengths, input.size(-1))
        if self.use_triangular_mask:
            if self.triangular_mask is None or output.size(0) > self.triangular_mask.size(0):
                self.triangular_mask = get_attention_mask(output.size(0)).type_as(input)
            square_mask = self.triangular_mask.narrow(0, 0, output.size(0)).narrow(1, 0, output.size(0))
        else:
            square_mask = None

        for layer in self.layers:
            output = layer(output, src_mask=square_mask, src_key_padding_mask=padding_mask)

        if self.final_norm is not None and final_norm:
            output = self.final_norm(output)

        output.transpose_(0, 1).transpose_(1, 2)  # (T, B, F) -> (B, T, F) -> (B, F, T)

        return padding_mask, output
