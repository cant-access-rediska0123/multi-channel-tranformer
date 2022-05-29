from data.text.text import MultiSpeakerTexts
from data.text.text_processor.text_processor import TextProcessor
from data.trans import Trans
from data.utterance.utterance import RawMultiSpeakerUtterance


class MultiSpeakerTextProcessor(Trans[RawMultiSpeakerUtterance, RawMultiSpeakerUtterance]):
    def __init__(self, **kwargs):
        self._text_processor = TextProcessor(**kwargs)

    def process_text(self, text: MultiSpeakerTexts) -> MultiSpeakerTexts:
        return MultiSpeakerTexts([self._text_processor.process_text(t) for t in text])

    def apply(self, utt: RawMultiSpeakerUtterance, **kwargs) -> RawMultiSpeakerUtterance:
        return RawMultiSpeakerUtterance(
            self.process_text(utt.text),
            utt.audio, utt.uuid, utt.speaker_id)
