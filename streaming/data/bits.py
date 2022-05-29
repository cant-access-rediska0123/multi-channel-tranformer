from dataclasses import dataclass
from typing import Generic, List

from data.audio.spectrogram import Spectrogram
from data.text.text import MultiSpeakerTexts, Text
from data.utterance.utterance import U, Utterance


@dataclass
class SampleInfo:
    total_ms: float
    orig_uuid: str
    orig_texts: List[Text]


@dataclass
class BitInfo:
    start_ms: float
    end_ms: float
    sample_info: SampleInfo


class BitUtterance(Generic[U]):
    def __init__(self, audio: U, bit_info: BitInfo):
        self._audio = audio
        self._bit_info = bit_info

    @property
    def audio(self) -> U:
        return self._audio

    @property
    def info(self) -> BitInfo:
        return self._bit_info


class MelBitUtterance(BitUtterance[Spectrogram]):
    pass


class SplitMultiSpeakerMelUtterance(Utterance[MultiSpeakerTexts, List[MelBitUtterance]]):
    pass
