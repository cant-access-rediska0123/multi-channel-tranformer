from factory.factory import Factory
from models.text_decoder.ctc_text_decoder import CtcTextDecoder
from models.text_decoder.sentencepiece_text_decoder import DiarizationTransformerTextDecoder, \
    SequentialDiarizationTransformerTextDecoder
from models.text_decoder.text_decoder import TextDecoder

Factory.register(TextDecoder, {
    'ctc_text_decoder': CtcTextDecoder,
    'sentencepiece_text_decoder': DiarizationTransformerTextDecoder,
    'sequential_sentencepiece_text_decoder': SequentialDiarizationTransformerTextDecoder,
})
