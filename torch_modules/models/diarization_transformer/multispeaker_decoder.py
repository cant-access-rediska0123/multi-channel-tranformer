from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from data.text.dictionary import SentencePieceDictionary
from torch_modules.common.linear import Linear
from torch_modules.models.transformer.encoder_decoder import DecoderResult, EncoderResult, create_layer
from torch_modules.models.transformer.transformer_decoder import TransformerDecoder


def _flatten_speakers(texts: torch.Tensor, text_lengths: torch.Tensor):
    batch_size, _, max_length, _ = texts.shape
    texts = texts.transpose(1, 2) \
        .reshape((batch_size, max_length, -1))  # (B, Sp, T, E) -> (B, T, Sp, E) -> (B, T, Sp * E)
    if text_lengths is not None:
        text_lengths = torch.max(text_lengths, 1).values  # (B, Sp) -> (B)
    return texts, text_lengths


class MultiSpeakerDecoder(nn.Module):
    def __init__(self, decoder_config: List[dict], input_dim: int, dictionary: SentencePieceDictionary,
                 speakers_num: int):
        super().__init__()
        self.speakers_num = speakers_num

        self._layers = nn.ModuleList()
        self._text_layers = nn.ModuleList()
        self._final_text_layer = Linear(speakers_num * input_dim, input_dim)

        self._dict_sz = len(dictionary)
        self._dictionary = dictionary
        tokens_dim = None
        for i in range(len(decoder_config)):
            layer_config = decoder_config[i].copy()
            layer_name = layer_config.pop("name")
            layer_config["dictionary"] = dictionary

            if layer_name in ("embedding", "positional_encoding"):
                self._text_layers.extend(create_layer(layer_name, layer_config, tokens_dim))
                tokens_dim = self._text_layers[-1].output_dim()
            else:
                self._layers.extend(
                    create_layer(layer_name, layer_config, input_dim, output_dim=self._dict_sz * speakers_num))
                input_dim = self._layers[-1].output_dim()

    def num_weights(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_result: EncoderResult, texts: Tensor, text_lengths: Optional[Tensor]) -> DecoderResult:
        batch_size, speakers_num, max_length = texts.size(0), texts.size(1), texts.size(2)

        texts = texts.reshape((-1, max_length))
        for layer in self._text_layers:
            texts = layer(texts)
        texts = texts.reshape((batch_size, speakers_num, max_length, texts.size(2)))
        texts, text_lengths = _flatten_speakers(texts, text_lengths)  # (B, T, Sp * E), (B)
        texts = self._final_text_layer(texts)  # (B, T, Sp * E) -> (B, T, E)

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

        output = output.reshape((batch_size, max_length, speakers_num, -1))
        output = output.transpose(1, 2)  # (B, T, Sp, *) -> (B, Sp, T, *)
        return DecoderResult(output=output)
