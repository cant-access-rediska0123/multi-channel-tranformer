import abc

from data.text.text import Text, TextTokenizer, TokenizedText, TokensDecoder


class Dictionary(abc.ABC):

    @property
    def tokenizer(self) -> TextTokenizer:
        return Dictionary._Encoder(self)

    @property
    def decoder(self) -> TokensDecoder:
        return Dictionary._Decoder(self)

    @abc.abstractmethod
    def encode(self, sample: Text) -> TokenizedText:
        pass

    @abc.abstractmethod
    def decode(self, sample: TokenizedText) -> Text:
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    def __call__(self, sample):
        if issubclass(type(sample), Text):
            return self.encode(sample)
        elif issubclass(type(sample), TokenizedText):
            return self.decode(sample)
        else:
            assert False

    class _Decoder(TokensDecoder):

        def apply(self, text: TokenizedText, **kwargs) -> Text:
            return self._owner.decode(text)

        def __init__(self, owner):
            self._owner = owner

    class _Encoder(TextTokenizer):

        def apply(self, text: Text, **kwargs) -> TokenizedText:
            return self._owner.encode(text)

        def __init__(self, owner):
            self._owner = owner
