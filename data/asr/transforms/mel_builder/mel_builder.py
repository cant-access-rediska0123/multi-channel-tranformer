from typing import Tuple, TypeVar

from data.asr.transforms.augmentations.chain_augmntation import SampledChainAugmentation, Trans
from data.asr.transforms.augmentations.spectrogram_augmentations import SpectrogramAugmetation
from data.asr.transforms.augmentations.wave_augmentations import WaveAugmentation
from data.asr.transforms.spectrogram_builder.spectrogram_builder import SpectrogramBuilder
from data.audio.spectrogram import Spectrogram
from data.audio.wave import Wave
from data.text.dictionary import Dictionary
from data.text.text import Text, TokenizedText
from data.utterance.utterance import Utterance
from factory.factory import make_instance

TextInp = TypeVar('TextInp')
TextOutp = TypeVar('TextOutp')
AudioInp = TypeVar('AudioInp')


class MelBuilder(Trans[Utterance[TextInp, Spectrogram], Utterance[TextOutp, Spectrogram]]):
    def __init__(self,
                 audio_transform: Trans[Wave, Wave],
                 spec_builder: SpectrogramBuilder,
                 mel_transform: Trans[Spectrogram, Spectrogram],
                 text_transform: Trans[TextInp, TextOutp]):
        self._audio_transform = audio_transform
        self._spec_builder = spec_builder
        self._mel_transform = mel_transform
        self._text_transform = text_transform

    def apply(self, sample: Utterance[TextInp, Wave], **kwargs) -> Utterance[TextOutp, Spectrogram]:
        return Utterance(self._text_transform(sample.text),
                         self._mel_transform(self._spec_builder(self._audio_transform(sample.audio))),
                         sample.uuid, sample.speaker_id)


def parse_pipeline(pipeline) -> Tuple:
    wave_augmentations = SampledChainAugmentation.create(
        WaveAugmentation, pipeline.get("wave_augmentations", []))
    mel_augmentations = SampledChainAugmentation.create(
        SpectrogramAugmetation, pipeline.get("spec_augmentations", []))
    spec_builder = make_instance(SpectrogramBuilder, pipeline["mel_calcer"])
    return wave_augmentations, mel_augmentations, spec_builder


class MelUtteranceBuilder(MelBuilder[Text, TokenizedText]):
    @staticmethod
    def create(pipeline,
               dictionary: Dictionary):
        wave_augmentations, mel_augmentations, spec_builder = parse_pipeline(pipeline)
        return MelBuilder(audio_transform=wave_augmentations,
                          spec_builder=spec_builder,
                          mel_transform=mel_augmentations,
                          text_transform=dictionary.tokenizer)
