from typing import List

from data.text.dictionary.ctc_dictionary import CtcDictionary
from data.text.text import Text, TokenizedText
from models.text_decoder.text_decoder import TextDecoder


class CtcTextDecoder(TextDecoder[CtcDictionary]):

    def __init__(self, dictionary: CtcDictionary):
        super().__init__(dictionary)

    def decode(self, predictions: List[TokenizedText]) -> List[Text]:
        blank_id = self.dictionary.blank_id()
        hypotheses = []
        # iterate over batch
        for prediction in predictions:
            # CTC decoding procedure
            decoded_prediction = []
            previous = blank_id
            for p in prediction.tokens:
                if (p != previous or previous == blank_id) and p != blank_id:
                    decoded_prediction.append(p)
                previous = p
            hypothesis = self.dictionary.decode(TokenizedText(decoded_prediction))
            hypotheses.append(hypothesis)
        return hypotheses
