from data.text.dictionary.ctc_dictionary import CtcLettersDict
from data.text.dictionary.dictionary import Dictionary
from data.text.dictionary.sentencepiece_dictionary import SentencePieceDictionary
from factory.factory import Factory

Factory.register(Dictionary, {
    "ctc_dictionary": CtcLettersDict,
    "sentencepiece_dictionary": SentencePieceDictionary
})
