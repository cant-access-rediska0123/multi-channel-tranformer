from abc import ABC
from typing import Generic, Optional, TypeVar

from data.audio.spectrogram import Spectrogram
from data.audio.wave import Wave
from data.text.text import MultiSpeakerTexts, MultiSpeakerTokenizedTexts, Text, TokenizedText

T = TypeVar('T')
U = TypeVar('U')


class Utterance(Generic[T, U], ABC):

    def __init__(self,
                 text: T,
                 audio: U,
                 uuid: str,
                 speaker_id: Optional[str] = None):
        self._text = text
        self._audio = audio
        self._uuid = uuid
        self._speaker_id = speaker_id

    @property
    def text(self) -> T:
        return self._text

    @property
    def uuid(self) -> str:
        return self._uuid

    @property
    def audio(self) -> U:
        return self._audio

    @property
    def speaker_id(self) -> Optional[str]:
        return self._speaker_id


class RawUtterance(Utterance[Text, Wave]):
    pass


class RawMultiSpeakerUtterance(Utterance[MultiSpeakerTexts, Wave]):
    pass


class MelUtterance(Utterance[TokenizedText, Spectrogram]):
    pass


class MelMultiSpeakerUtterance(Utterance[MultiSpeakerTokenizedTexts, Spectrogram]):
    pass
