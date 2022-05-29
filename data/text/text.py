import abc
from typing import List, NewType

from data.trans import Trans

Text = NewType("Text", str)
MultiSpeakerTexts = NewType("Text", List[Text])


class TokenizedText:

    def __init__(self,
                 tokenized_text: List[int]):
        self._tokenized_text: List[int] = tokenized_text

    @property
    def tokens(self) -> List[int]:
        return self._tokenized_text

    def __str__(self) -> str:
        tokens_str = ",".join(map(str, self._tokenized_text))
        return f"[{tokens_str}]"

    def __repr__(self) -> str:
        if len(self.tokens) < 5:
            return str(self)
        return f"[{self._tokenized_text[0]},  {self._tokenized_text[1]}, ..., {self._tokenized_text[-1]}]"

    def __len__(self):
        return len(self.tokens)


MultiSpeakerTokenizedTexts = NewType("Text", List[TokenizedText])


class TextTokenizer(Trans[Text, TokenizedText]):

    @abc.abstractmethod
    def apply(self, text: Text, **kwargs) -> TokenizedText:
        pass


class TokensDecoder(Trans[TokenizedText, Text]):

    @abc.abstractmethod
    def apply(self, text: TokenizedText, **kwargs) -> Text:
        pass
