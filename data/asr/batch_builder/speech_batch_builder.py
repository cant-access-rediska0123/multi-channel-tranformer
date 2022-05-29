from typing import List, Tuple

import torch
from torch import Tensor

from data.asr.batch.speech_batch import SpeechBatch
from data.asr.batch_builder.audio_duration_calcer import AudioDurationCalcer
from data.asr.batch_builder.batch_builder import BatchBuilder
from data.audio.spectrogram import Spectrogram
from data.text.text import TokenizedText
from data.utterance.utterance import MelUtterance


def _align(size, pad_to) -> int:
    return ((size + pad_to - 1) // pad_to) * pad_to


def build_features(specs: List[Spectrogram], pad_to, features_pad: float = 0.0) -> Tuple[Tensor, Tensor]:
    batch_size = len(specs)

    features_dim = specs[0].dim
    max_features_length = _align(max(len(s) for s in specs), pad_to=pad_to)

    batched_features = torch.full((batch_size, max_features_length, features_dim),
                                  fill_value=features_pad, dtype=torch.float32)
    features_lengths = []
    for i, s in enumerate(specs):
        features_length = s.length
        features_lengths.append(features_length)
        batched_features[i].narrow(0, 0, features_length).copy_(torch.tensor(s.spectrogram))

    features = batched_features.permute([0, 2, 1])
    features_lengths = torch.tensor(features_lengths)
    return features, features_lengths


def _build_tokens(texts: List[TokenizedText], text_pad: int):
    max_tokens_length = max(len(t) for t in texts)
    tokens = torch.full((len(texts), max_tokens_length),
                        dtype=torch.int32, fill_value=text_pad)
    tokens_lengths = []
    for i, text in enumerate(texts):
        tokens_length = len(text.tokens)
        tokens[i].narrow(0, 0, tokens_length).copy_(torch.tensor(text.tokens, dtype=torch.int32))
        tokens_lengths.append(tokens_length)
    return tokens, torch.tensor(tokens_lengths)


class SpeechBatchBuilder(BatchBuilder[MelUtterance]):

    def __init__(self,
                 pad_to=1,
                 audio_duration_for_units=False,
                 features_pad=0.0,
                 text_pad=0):
        super().__init__([AudioDurationCalcer()] if audio_duration_for_units else None)
        self._pad_to = pad_to
        self._features_pad = features_pad
        self._text_pad = text_pad

    def _build(self, batch: List[MelUtterance]) -> SpeechBatch:
        features, features_lengths = build_features([b.audio for b in batch], self._pad_to, self._features_pad)
        tokens, tokens_lengths = _build_tokens([b.text for b in batch], self._text_pad)

        return SpeechBatch(features,
                           features_lengths,
                           tokens,
                           tokens_lengths,
                           [sample.text for sample in batch],
                           [sample.uuid for sample in batch])
