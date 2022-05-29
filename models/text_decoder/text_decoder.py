import abc
from typing import Generic, List, TypeVar

import torch

from data.text.dictionary import Dictionary
from data.text.text import MultiSpeakerTexts, TokenizedText

T = TypeVar('T')


class TextDecoder(Generic[T]):
    def __init__(self,
                 dictionary: Dictionary):
        self.__dictionary = dictionary

    @property
    def dictionary(self) -> T:
        return self.__dictionary

    @abc.abstractmethod
    def decode(self, predictions: List[TokenizedText]) -> MultiSpeakerTexts:
        pass

    def decode_tensor(self, predictions: torch.Tensor) -> MultiSpeakerTexts:
        texts = [TokenizedText(p.tolist()) for p in predictions.cpu()]
        return self.decode(texts)

    def __call__(self, predictions) -> MultiSpeakerTexts:
        if isinstance(predictions, torch.Tensor):
            return self.decode_tensor(predictions)
        else:
            return self.decode(predictions)
