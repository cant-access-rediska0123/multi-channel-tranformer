from typing import List

import torch

from data.asr.batch.speech_batch import SpeechBatch
from data.asr.batch_builder.audio_duration_calcer import AudioDurationCalcer
from data.asr.batch_builder.batch_builder import BatchBuilder
from data.asr.batch_builder.speech_batch_builder import build_features
from data.text.text import TokenizedText
from data.utterance.utterance import MelMultiSpeakerUtterance


def _build_multispeaker_tokens(texts: List[List[TokenizedText]], text_pad: int):
    max_tokens_length = max([len(t) for text in texts for t in text], default=1)
    tokens = torch.full((len(texts), len(texts[0]), max_tokens_length),
                        dtype=torch.int32, fill_value=text_pad)
    tokens_lengths = torch.zeros((len(texts), len(texts[0])), dtype=torch.int32)
    for i, text in enumerate(texts):
        for speaker_id, speaker_text in enumerate(text):
            tokens_length = len(speaker_text.tokens)
            tokens[i, speaker_id].narrow(0, 0, tokens_length).copy_(
                torch.tensor(speaker_text.tokens, dtype=torch.int32))
            tokens_lengths[i, speaker_id] = tokens_length
    return tokens, tokens_lengths


class MultiSpeakerSpeechBatchBuilder(BatchBuilder[MelMultiSpeakerUtterance]):

    def __init__(self,
                 pad_to=1,
                 audio_duration_for_units=False,
                 features_pad=0.0,
                 text_pad=0):
        super().__init__([AudioDurationCalcer()] if audio_duration_for_units else None)
        self._pad_to = pad_to
        self._features_pad = features_pad
        self._text_pad = text_pad

    def _build(self, batch: List[MelMultiSpeakerUtterance]) -> SpeechBatch:
        features, features_lengths = build_features([b.audio for b in batch], self._pad_to, self._features_pad)
        tokens, tokens_lengths = _build_multispeaker_tokens([b.text for b in batch], self._text_pad)

        return SpeechBatch(features,
                           features_lengths,
                           tokens,
                           tokens_lengths,
                           [sample.text for sample in batch],
                           [sample.uuid for sample in batch])
