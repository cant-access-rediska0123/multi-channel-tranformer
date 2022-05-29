import abc

from data.audio.spectrogram import Spectrogram
from data.audio.wave import Wave
from data.trans import Trans


class SpectrogramBuilder(Trans[Wave, Spectrogram]):

    @abc.abstractmethod
    def apply(self, wave: Wave, **kwargs) -> Spectrogram:
        pass

    @abc.abstractmethod
    def dim(self):
        pass
