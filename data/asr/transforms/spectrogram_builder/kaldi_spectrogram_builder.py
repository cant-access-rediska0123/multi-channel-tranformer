import numpy as np

from data.asr.transforms.spectrogram_builder.spectrogram_builder import Spectrogram, SpectrogramBuilder, Wave


class KaldiFeaturesSpectrogram(SpectrogramBuilder):

    def __init__(self, **kwargs):
        config = dict(kwargs)
        if "normalize" in config:
            self._normalize = config.get("normalize")
            config.pop("normalize")
        else:
            self._normalize = False
        self._config = config
        self._mel_features_calcer = None
        self._sample_rate = config["sample-frequency"]
        self._frame_shift = config["frame-shift"]
        self._frame_size = config["frame-length"]
        self._mel_dim = config["num-mel-bins"]

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        # replace the `value` key (now an EnumValue instance), with it's index:
        state.pop("_mel_features_calcer")
        # what we return here will be stored in the pickle
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)
        self._mel_features_calcer = None

    def apply(self, sample: Wave, **kwargs) -> Spectrogram:
        assert sample.sample_rate == self._sample_rate
        if self._mel_features_calcer is None:
            import kaldi_features
            self._mel_features_calcer = kaldi_features.KaldiFeaturesCalcer(self._config)
        result = self._mel_features_calcer.compute(sample.data)
        if self._normalize:
            m = np.mean(result)
            sd = np.std(result)
            result = (result - m) / (sd + 1e-9)
        return Spectrogram(result,
                           self._frame_shift,
                           self._frame_size,
                           orig_audio=sample)

    def dim(self):
        return self._mel_dim
