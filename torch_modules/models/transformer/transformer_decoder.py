from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.text.dictionary import SentencePieceDictionary
from torch_modules.common.utils import get_activation_fn, get_attention_mask, get_clones, get_padding_mask


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)[0]
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.dropout3(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, model_definition: dict, dictionary: SentencePieceDictionary,
                 output_dim: Optional[int] = None, **kwargs):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model=model_definition["d_model"],
                                                num_heads=model_definition["num_heads"],
                                                dim_feedforward=model_definition["dim_feedforward"],
                                                dropout=model_definition["dropout"],
                                                activation=model_definition["activation"],
                                                normalize_before=model_definition["normalize_before"])
        self.num_layers = model_definition["num_layers"]
        self.d_model = model_definition["d_model"]
        self.layers = get_clones(decoder_layer, self.num_layers)
        self.final_norm = None
        if model_definition["final_norm"]:
            self.final_norm = nn.LayerNorm(self.d_model)
        self._output_dim = output_dim if output_dim else len(dictionary)
        self.final_linear = nn.Linear(self.d_model, self._output_dim)

    def input_dim(self) -> int:
        return self.d_model

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor, memory: torch.Tensor,
                memory_square_mask: Optional[torch.Tensor] = None,
                memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_square_mask = get_attention_mask(input.size(1)).type_as(input)
        input_padding_mask = None
        if input_lengths is not None:
            input_padding_mask = get_padding_mask(input_lengths)
        output = input.transpose(0, 1)  # (B, T, F) -> (T, B, F)
        memory = memory.transpose(1, 2).transpose(0, 1)

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=input_square_mask,
                           memory_mask=memory_square_mask,
                           tgt_key_padding_mask=input_padding_mask,
                           memory_key_padding_mask=memory_padding_mask)

        if self.final_norm is not None:
            output = self.final_norm(output)

        output = output.transpose(0, 1)  # (T, B, F) -> (B, T, F)
        output = self.final_linear(output)  # (B, T, F) -> (B, T, vocab_size)

        return output
