import abc

import numpy as np

from data.trans import Trans


class Wave:

    def __init__(self,
                 sample_rate: int,
                 data: np.ndarray):
        self._sample_rate = sample_rate
        self._data = data

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def data(self):
        return self._data

    @property
    def duration_secs(self) -> float:
        return len(self._data) * 1.0 / self.sample_rate


class WavTransform(Trans[Wave, Wave]):

    @abc.abstractmethod
    def apply(self, sample: Wave, **kwargs) -> Wave:
        pass
