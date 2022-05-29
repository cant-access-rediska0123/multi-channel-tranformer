import os

import sentencepiece as spm

from data.text.dictionary.dictionary import Dictionary
from data.text.text import Text, TokenizedText


class SentencePieceDictionary(Dictionary):
    def __init__(self, dict_path: str):
        with open(dict_path, "rb") as f:
            self._serialized_dict = f.read()
        self._sp_model = spm.SentencePieceProcessor()
        self._sp_model.load_from_serialized_proto(self._serialized_dict)
        self._dict_file = os.path.basename(dict_path)

    def encode(self, sample: Text) -> TokenizedText:
        ids = [self.bos_id()] + self._sp_model.encode_as_ids(sample) + [self.eos_id()]
        return TokenizedText(ids)

    def decode(self, sample: TokenizedText) -> Text:
        # ignore all ids after eos_id
        n = next((i for i, x in enumerate(sample.tokens) if x == self.eos_id()), len(sample))
        text = self._sp_model.decode_ids(sample.tokens[:n])
        return text

    def unk_id(self) -> int:
        return self._sp_model.unk_id()

    def pad_id(self) -> int:
        id = self._sp_model.pad_id()
        assert id != self.unk_id(), "'pad' token is not defined"
        return id

    def bos_id(self) -> int:
        id = self._sp_model.bos_id()
        assert id != self.unk_id(), "'bos' token is not defined"
        return id

    def eos_id(self) -> int:
        id = self._sp_model.eos_id()
        assert id != self.unk_id(), "'eos' token is not defined"
        return id

    def sil_id(self) -> int:
        id = self._sp_model.piece_to_id("<SIL>")
        assert id != self.unk_id(), "'sil' token is not defined"
        return id

    def space_id(self) -> int:
        return self._sp_model.piece_to_id("▁")

    def __len__(self):
        return len(self._sp_model)
