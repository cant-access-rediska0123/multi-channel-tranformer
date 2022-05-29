from abc import ABC
from typing import Optional

import numpy as np

from data.audio.wave import Wave


class Spectrogram(ABC):

    def __init__(self,
                 spectrogram: np.array,
                 shift_ms: int,
                 frame_size_ms: int,
                 orig_audio: Optional[Wave] = None):
        self._spectrogram = spectrogram
        self._frame_shift_ms = shift_ms
        self._frame_size_ms = frame_size_ms
        self._orig_audio = orig_audio

    @property
    def spectrogram(self) -> np.array:
        return self._spectrogram

    @property
    def duration_secs(self):
        return self.length * self._frame_shift_ms / 1000.

    @property
    def dim(self):
        return self._spectrogram.shape[1]

    @property
    def orig_audio(self):
        return self._orig_audio

    @property
    def length(self):
        return self._spectrogram.shape[0]

    @property
    def frame_shift_ms(self):
        return self._frame_shift_ms

    @property
    def frame_size_ms(self):
        return self._frame_size_ms

    def __len__(self):
        return self.length
