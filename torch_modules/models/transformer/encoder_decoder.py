from dataclasses import dataclass
from typing import List, Optional

import torch.nn as nn
from torch import Tensor

from data.text.dictionary import SentencePieceDictionary
from torch_modules.common.coalescence import Coalescence
from torch_modules.common.conv import Conv
from torch_modules.common.embedding import Embedding
from torch_modules.common.linear import Linear
from torch_modules.common.positional_encoding import PositionalEncoding
from torch_modules.models.transformer.transformer_decoder import TransformerDecoder
from torch_modules.models.transformer.transformer_encoder import TransformerEncoder


def create_layer(name: str, config: dict, input_dim: int, output_dim: Optional[int] = None) -> List[nn.Module]:
    layer_types = {
        "embedding": Embedding,
        "positional_encoding": PositionalEncoding,
        "coalescence": Coalescence,
        "transformer_encoder": TransformerEncoder,
        "transformer_decoder": TransformerDecoder,
        "conv": Conv,
    }

    layers = []

    expected_input_dim = config.get("input_dim")
    # if the layer takes an arbitrary dimension, specify the current input dimension in case the layer needs it
    if expected_input_dim is None:
        config["input_dim"] = input_dim
    if output_dim is not None:
        config["output_dim"] = output_dim
    # if the layer works with a specific dimension, add a linear layer to match the dimensions
    elif expected_input_dim is not None and input_dim is not None and input_dim != expected_input_dim:
        layers.append(Linear(input_dim, expected_input_dim))

    layers.append(layer_types[name](**config))

    return layers


@dataclass
class EncoderResult:
    output: Tensor
    encoded_lengths: Tensor
    transformer_encoder_padding_mask: Tensor


class Encoder(nn.Module):
    def __init__(self, encoder_config: List[dict], input_dim: int, dictionary: SentencePieceDictionary):
        super().__init__()

        self._input_dim = input_dim
        self._layers = nn.ModuleList()

        for i in range(len(encoder_config)):
            layer_config = encoder_config[i].copy()
            layer_name = layer_config.pop("name")
            layer_config["dictionary"] = dictionary
            self._layers.extend(create_layer(layer_name, layer_config, input_dim))
            input_dim = self._layers[-1].output_dim()

        self._output_dim = self._layers[-1].output_dim()

    def input_dim(self) -> int:
        return self._input_dim

    def output_dim(self) -> int:
        return self._output_dim

    def num_weights(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input: Tensor, input_lengths: Optional[Tensor]) -> EncoderResult:
        output = input
        transformer_encoder_padding_mask = None
        for layer in self._layers:
            if isinstance(layer, (Linear, PositionalEncoding, Conv)):
                output = layer(output)
            elif isinstance(layer, (Coalescence)):
                output, input_lengths = layer(output, input_lengths)
            elif isinstance(layer, TransformerEncoder):
                transformer_encoder_padding_mask, output = layer(output, input_lengths)
            else:
                raise Exception(f"Unknown encoder layer: {layer}")

        return EncoderResult(output, input_lengths, transformer_encoder_padding_mask)


@dataclass
class DecoderResult:
    output: Tensor


class Decoder(nn.Module):
    def __init__(self, decoder_config: List[dict], input_dim: int, dictionary: SentencePieceDictionary):
        super().__init__()

        self._layers = nn.ModuleList()
        self._text_layers = nn.ModuleList()

        tokens_dim = None
        for i in range(len(decoder_config)):
            layer_config = decoder_config[i].copy()
            layer_name = layer_config.pop("name")
            layer_config["dictionary"] = dictionary

            if layer_name in ("embedding", "positional_encoding"):
                self._text_layers.extend(create_layer(layer_name, layer_config, tokens_dim))
                tokens_dim = self._text_layers[-1].output_dim()
            else:
                self._layers.extend(create_layer(layer_name, layer_config, input_dim))
                input_dim = self._layers[-1].output_dim()

    def num_weights(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_result: EncoderResult, texts: Tensor, text_lengths: Optional[Tensor]) -> DecoderResult:
        for layer in self._text_layers:
            texts = layer(texts)

        output = encoder_result.output
        for layer in self._layers:
            if isinstance(layer, Linear):
                output = layer(output)
            elif isinstance(layer, TransformerDecoder):
                output = layer(input=texts,
                               input_lengths=text_lengths,
                               memory=output,
                               memory_padding_mask=encoder_result.transformer_encoder_padding_mask)
            else:
                raise Exception(f"Unknown decoder layer: {layer}")

        return DecoderResult(output)
