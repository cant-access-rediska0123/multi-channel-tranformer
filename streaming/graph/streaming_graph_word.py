import numpy as np

from aligner.aligned_word import TimeAlignedWord
from data.text.text import Text


class StreamingGraphWord(TimeAlignedWord):
    def __init__(self, text: Text, start_ms: float, end_ms: float, score: int, precision: int = 50):
        super().__init__(text, start_ms, end_ms)
        self.score = score
        self._hash = hash((text, round(start_ms * precision), round(end_ms * precision)))

    def __eq__(self, other):
        return self.text == other.text and \
               np.isclose(self.start_ms, other.start_ms) and \
               np.isclose(self.end_ms, other.end_ms)

    def __hash__(self):
        return self._hash

    def __str__(self):
        return f'{self.text}: {self.start_ms}-{self.end_ms} ({self.score})'
