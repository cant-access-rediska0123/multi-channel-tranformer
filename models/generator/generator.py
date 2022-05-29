import abc
from typing import Generic, List, TypeVar

from data.asr.batch.speech_batch import SpeechBatch
from data.text.text import MultiSpeakerTexts, Text

BatchType = TypeVar('BatchType')
OutpType = TypeVar('OutpType')


class Generator(Generic[BatchType, OutpType]):
    @abc.abstractmethod
    def __call__(self, batch: BatchType) -> OutpType:
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class TextGenerator(abc.ABC, Generator[SpeechBatch, List[Text]]):
    pass


class DiarizationTextGenerator(abc.ABC, Generator[SpeechBatch, List[MultiSpeakerTexts]]):
    pass
