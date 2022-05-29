from typing import List

import numpy as np
import torch

from data.asr.batch.speech_batch import SpeechBatch
from data.text.dictionary.ctc_dictionary import CtcDictionary
from data.text.text import Text, TokenizedText
from models.generator.generator import TextGenerator
from models.text_decoder.ctc_text_decoder import CtcTextDecoder


class CtcGenerator(TextGenerator):
    def __init__(self, ctc_model: torch.nn.Module, dictionary: CtcDictionary):
        self._ctc_model = ctc_model
        self._dictionary = dictionary
        self._text_decoder = CtcTextDecoder(dictionary)

    def generate_from_logits(self, ctc_logits: np.ndarray) -> List[Text]:
        tokens = ctc_logits.argmax(axis=2)
        return self._text_decoder.decode([TokenizedText(tokens) for tokens in tokens])

    def __call__(self, batch: SpeechBatch) -> List[Text]:
        with torch.no_grad():
            ctc_logits = self._ctc_model(batch.features).cpu().numpy()  # (B, F, T)
        return self.generate_from_logits(ctc_logits)

    def __str__(self) -> str:
        return 'CtcGenerator'
