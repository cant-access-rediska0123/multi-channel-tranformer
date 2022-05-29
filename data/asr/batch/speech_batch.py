from typing import List

import torch
from torch import Tensor, nn

from data.asr.batch_builder.batch_builder import Batch
from data.utterance.utterance import MelUtterance


def clone_tensor_to_cuda(tensor):
    return tensor.clone() if tensor.is_cuda else tensor.cuda()


class SpeechBatch(Batch[MelUtterance], nn.Module):
    def __init__(self,
                 features: Tensor,
                 features_lengths: Tensor,
                 tokens: Tensor,
                 tokens_lengths: Tensor,
                 orig_texts: List,
                 orig_uuids: List[str]):
        super().__init__()
        nn.Module.__init__(self)
        self.register_buffer("features", features)
        self.register_buffer("features_lengths", features_lengths)
        self.register_buffer("tokens", tokens.long())
        self.register_buffer("tokens_lengths", tokens_lengths.long())
        self._batch_size = tokens.shape[0]
        self._orig_texts = orig_texts
        self._orig_uuids = orig_uuids

    def __len__(self):
        return self._batch_size

    @property
    def texts(self) -> List:
        return self._orig_texts

    @property
    def orig_uuids(self) -> List[str]:
        return self._orig_uuids

    @property
    def device(self) -> torch.device:
        return self.features.device

    def clone_to_gpu(self):
        return SpeechBatch(clone_tensor_to_cuda(self.features),
                           clone_tensor_to_cuda(self.features_lengths),
                           clone_tensor_to_cuda(self.tokens),
                           clone_tensor_to_cuda(self.tokens_lengths),
                           self._orig_texts,
                           self._orig_uuids)
