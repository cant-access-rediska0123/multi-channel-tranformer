from typing import List

from data.asr.transforms.mel_builder.mel_builder import MelBuilder, parse_pipeline
from data.text.dictionary import Dictionary
from data.text.text import MultiSpeakerTexts, TextTokenizer, TokenizedText
from data.trans import Trans


class MelMultiSpeakerUtteranceBuilder(MelBuilder[MultiSpeakerTexts, List[TokenizedText]]):
    class _MultiSpeakerTokenizer(Trans[MultiSpeakerTexts, List[TokenizedText]]):
        def __init__(self, tokenizer: TextTokenizer):
            self._tokenizer = tokenizer

        def apply(self, sample: MultiSpeakerTexts, **kwargs) -> List[TokenizedText]:
            return [self._tokenizer(s) for s in sample]

    @staticmethod
    def create(pipeline,
               dictionary: Dictionary):
        wave_augmentations, mel_augmentations, spec_builder = parse_pipeline(pipeline)
        return MelBuilder(audio_transform=wave_augmentations,
                          spec_builder=spec_builder,
                          mel_transform=mel_augmentations,
                          text_transform=MelMultiSpeakerUtteranceBuilder._MultiSpeakerTokenizer(dictionary.tokenizer))
