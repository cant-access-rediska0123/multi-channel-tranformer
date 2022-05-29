import abc
from typing import List

from data.text.dictionary.dictionary import Dictionary
from data.text.text import Text, TokenizedText


class CtcDictionary(Dictionary):

    @abc.abstractmethod
    def space_id(self) -> int:
        pass

    @abc.abstractmethod
    def blank_id(self) -> int:
        pass

    @abc.abstractmethod
    def pad_id(self) -> int:
        pass

    @abc.abstractmethod
    def sil_id(self) -> int:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class CtcLettersDict(CtcDictionary):

    def __init__(self, letters: List[str]):
        self._letters = letters
        self._ids_to_letters = dict((i, letter) for i, letter in enumerate(self._letters))
        self._letters_to_ids = dict((letter, i) for i, letter in enumerate(self._letters))

        self._validate_vocabulary()

    def encode(self, text: Text) -> TokenizedText:
        ids = [self._letters_to_ids.get(x) for x in text]
        tokens = [i for i in ids if i is not None]
        return TokenizedText(tokens)

    def decode(self, tokenized_text: TokenizedText) -> Text:
        return Text("".join(self._ids_to_letters.get(i, "") for i in tokenized_text.tokens))

    def sil_id(self) -> int:
        return self._letters_to_ids[' ']

    def space_id(self) -> int:
        return self._letters_to_ids[' ']

    def pad_id(self) -> int:
        return self.blank_id()

    def blank_id(self) -> int:
        return len(self._letters) - 1

    def _validate_vocabulary(self):
        if self._letters[-1] != "|":
            raise Exception("last letter expected to be blank, got {}".format(self._letters[-1]))
        for letter in self._letters:
            if letter == " ":
                return
        raise Exception("no space in dictionary")

    def __len__(self):
        return len(self._letters)
