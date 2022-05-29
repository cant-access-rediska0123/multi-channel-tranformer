from data.text.dictionary import SentencePieceDictionary
from data.text.text import MultiSpeakerTokenizedTexts, TokenizedText
from data.trans import Trans
from data.utterance.utterance import MelMultiSpeakerUtterance


class FlattenTexts(Trans[MelMultiSpeakerUtterance, MelMultiSpeakerUtterance]):
    def __init__(self, dictionary: SentencePieceDictionary):
        self._dictionary = dictionary

    def apply(self, sample: MelMultiSpeakerUtterance, **kwargs) -> MelMultiSpeakerUtterance:
        concatenated_texts = [
            t
            for i, text in enumerate(sample.text)
            for t in text.tokens
            if i + 1 == len(sample.text) or t != self._dictionary.eos_id()]
        return MelMultiSpeakerUtterance(
            MultiSpeakerTokenizedTexts([TokenizedText(concatenated_texts)]),
            sample.audio,
            sample.uuid, sample.speaker_id)
