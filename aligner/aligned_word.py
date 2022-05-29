from dataclasses import dataclass
from typing import List, NewType

from data.text.text import Text


@dataclass
class FrameAlignedWord:
    text: Text
    start_frame: float = 0
    end_frame: float = 0


@dataclass
class TimeAlignedWord:
    text: Text
    start_ms: float = 0
    end_ms: float = 0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def __str__(self):
        return f'{self.text}: {self.start_ms}-{self.end_ms}'

    def __repr__(self):
        return str(self)


FrameAlignedWords = NewType('Alignment', List[List[FrameAlignedWord]])  # (speaker, word_id)
TimeAlignedWords = NewType('Alignment', List[List[TimeAlignedWord]])  # (speaker, word_id)
