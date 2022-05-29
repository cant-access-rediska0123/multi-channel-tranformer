import abc
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from aligner.time_ctc_aligner import TimeAlignedWords

LOG = logging.getLogger()


@dataclass
class RecognitionModelOutput:
    alignment: TimeAlignedWords
    ctc_logits: Optional[np.ndarray] = None


class RecognitionModel:
    @abc.abstractmethod
    def __call__(self, batches: Tuple) -> List[RecognitionModelOutput]:
        pass
