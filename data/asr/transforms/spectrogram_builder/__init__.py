from data.asr.transforms.spectrogram_builder.kaldi_spectrogram_builder import KaldiFeaturesSpectrogram
from data.asr.transforms.spectrogram_builder.librosa_spectrogram_builder import LibrosaMelSpectrogram
from data.asr.transforms.spectrogram_builder.spectrogram_builder import SpectrogramBuilder
from factory.factory import Factory

Factory.register(SpectrogramBuilder, {
    "librosa": LibrosaMelSpectrogram,
    "kaldi": KaldiFeaturesSpectrogram,
})
