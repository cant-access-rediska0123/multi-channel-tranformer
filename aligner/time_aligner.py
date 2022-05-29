import abc
from typing import List

import numpy as np

from aligner.aligned_word import TimeAlignedWords
from data.text.text import MultiSpeakerTexts


class TimeAligner(abc.ABC):
    def __call__(self, references: List[MultiSpeakerTexts], ctc_logits: np.ndarray) -> List[TimeAlignedWords]:
        pass
