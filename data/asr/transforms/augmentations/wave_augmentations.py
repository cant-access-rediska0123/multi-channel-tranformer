import logging
import random
from abc import ABC

import librosa
import numpy as np

from data.audio.wave import WavTransform, Wave

LOG = logging.getLogger()


class WaveAugmentation(WavTransform, ABC):

    def __init__(self, max_duration):
        self._max_duration = max_duration

    @property
    def max_duration(self):
        return self._max_duration


class SpeedAugmentation(WaveAugmentation):

    def __init__(self,
                 max_duration,
                 min_speed_rate=0.75,
                 max_speed_rate=1.25):
        super().__init__(max_duration)
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def apply(self, sample: Wave, **kwargs) -> Wave:
        # wave, sample_rate = audio_sample.wave
        wave_length = sample.duration_secs
        max_stretched_length = wave_length * self._max_rate
        max_rate = min(self.max_duration, max_stretched_length) / wave_length
        min_rate = min(max_rate - 1e-4, self._min_rate)
        speed_rate = random.Random().uniform(min_rate, max_rate)
        if speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")
        time_stretched_wave = librosa.effects.time_stretch(sample.data.astype(np.float32),
                                                           speed_rate)
        return Wave(sample.sample_rate,
                    time_stretched_wave)


class GainAugmentation(WaveAugmentation):

    def __init__(self,
                 max_duration,
                 min_gain_dbfs=-20,
                 max_gain_dbfs=20):
        super().__init__(max_duration)
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs

    def apply(self, sample: Wave, **kwargs) -> Wave:
        gain = random.Random().uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        return Wave(sample.sample_rate,
                    sample.data * (10. ** (gain / 20.)))
