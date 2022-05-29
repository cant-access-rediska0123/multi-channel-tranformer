import numpy as np

from data.asr.transforms.spectrogram_builder.spectrogram_builder import SpectrogramBuilder
from data.audio.spectrogram import Spectrogram
from data.audio.wave import Wave


class LibrosaMelSpectrogram(SpectrogramBuilder):
    def __init__(self,
                 **params):
        self._sample_rate = params["sample-frequency"]
        self._num_freq = params["num-freq"]
        self._min_frequency = params["min-frequency"]
        self._max_frequency = params["max-frequency"]
        self._num_mels = params["num-mel-bins"]

        self._n_fft = (self._num_freq - 1) * 2

        self._win_length = self._n_fft
        self._hop_length = self._n_fft // 4

        import librosa.core
        self._mel_basis = librosa.filters.mel(self._sample_rate,
                                              self._n_fft,
                                              n_mels=self._num_mels,
                                              fmin=self._min_frequency,
                                              fmax=self._max_frequency)

        self._frame_shift = self._hop_length * 1000 / self._sample_rate
        self._frame_size = self._win_length * 1000 / self._sample_rate

    def __stft(self, y):
        import librosa
        return librosa.stft(y,
                            n_fft=self._n_fft,
                            hop_length=self._hop_length,
                            win_length=self._win_length,
                            window='hann')

    def __linear_to_mel(self, s):
        return np.dot(self._mel_basis, s)

    def __amp_to_db(self, x):
        return np.log(np.maximum(1e-5, x))

    def __mel_spectrogram(self, wav):
        s = np.abs(self.__stft(wav))
        mels = self.__amp_to_db(self.__linear_to_mel(s))
        return mels

    def apply(self, wave: Wave, **kwargs) -> Spectrogram:
        return Spectrogram(self.__mel_spectrogram(wave.data),
                           self._frame_shift,
                           self._frame_size,
                           orig_audio=wave)

    def dim(self):
        return self._num_mels
