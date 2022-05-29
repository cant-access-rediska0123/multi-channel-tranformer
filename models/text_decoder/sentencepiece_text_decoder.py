from typing import List

from data.text.dictionary.sentencepiece_dictionary import SentencePieceDictionary
from data.text.text import MultiSpeakerTexts, Text, TokenizedText
from models.text_decoder.text_decoder import TextDecoder


class DiarizationTransformerTextDecoder(TextDecoder[SentencePieceDictionary]):
    def __init__(self, dictionary: SentencePieceDictionary):
        super().__init__(dictionary)

    def decode(self, texts: List[TokenizedText]) -> MultiSpeakerTexts:
        return MultiSpeakerTexts([self.dictionary.decode(t) for t in texts])


class SequentialDiarizationTransformerTextDecoder(TextDecoder[SentencePieceDictionary]):
    def __init__(self, dictionary: SentencePieceDictionary):
        super().__init__(dictionary)

    def decode(self, texts: List[TokenizedText]) -> List[Text]:
        assert len(texts) == 1
        text = texts[0]
        words = [[]]
        for t in text.tokens:
            if t == self.dictionary.bos_id():
                if len(words[-1]) > 0:
                    words.append([])
            elif t == self.dictionary.eos_id():
                break
            else:
                words[-1].append(t)
        return [self.dictionary.decode(TokenizedText(w)) for w in words]
