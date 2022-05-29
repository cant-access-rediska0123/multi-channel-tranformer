from typing import List

import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor

from data.asr.batch.batch import Batch
from data.asr.batch.speech_batch import clone_tensor_to_cuda
from data.asr.batch_builder.batch_builder import BatchBuilder
from data.asr.batch_builder.speech_batch_builder import build_features
from data.utterance.utterance import MelMultiSpeakerUtterance
from streaming.data.bits import BitInfo, MelBitUtterance


class BitsMultiSpeakerBatch(Batch[MelMultiSpeakerUtterance], LightningModule):
    def __init__(self,
                 features: Tensor,
                 features_lengths: Tensor,
                 bits_info: List[BitInfo]):
        nn.Module.__init__(self)
        super().__init__()
        self.register_buffer("features", features)
        self.register_buffer("features_lengths", features_lengths)
        self._batch_size = features.shape[0]
        self._bits_info = bits_info

    @property
    def info(self) -> List[BitInfo]:
        return self._bits_info

    def __len__(self):
        return self._batch_size

    def clone_to_gpu(self):
        return BitsMultiSpeakerBatch(clone_tensor_to_cuda(self.features),
                                     clone_tensor_to_cuda(self.features_lengths),
                                     self._bits_info)


class BitsBatchBuilder(BatchBuilder[MelBitUtterance]):
    def __init__(self,
                 pad_to=1,
                 features_pad=0.0):
        super().__init__(None)
        self._pad_to = pad_to
        self._features_pad = features_pad

    def _build(self, batch: List[MelBitUtterance]) -> BitsMultiSpeakerBatch:
        features, features_lengths = build_features([b.audio for b in batch], self._pad_to, self._features_pad)
        return BitsMultiSpeakerBatch(features, features_lengths, [b.info for b in batch])
