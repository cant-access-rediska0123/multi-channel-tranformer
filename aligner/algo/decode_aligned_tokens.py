import copy
from dataclasses import dataclass
from typing import List

from aligner.aligned_word import FrameAlignedWord
from data.text.dictionary.ctc_dictionary import CtcDictionary
from data.text.text import Text, TokenizedText


@dataclass
class _AlignedToken:
    token_id: int
    start_frame: int
    end_frame: int


def _delete_duplicate_tokens(tokens: List[_AlignedToken]) -> List[_AlignedToken]:
    res_tokens: List[_AlignedToken] = []
    for i, token in enumerate(tokens):
        if i == 0 or token.token_id != tokens[i - 1].token_id:
            res_tokens.append(token)
        else:
            res_tokens[-1].end_frame = i
    return res_tokens


def _delete_blank_ids(tokens: List[_AlignedToken], dictionary: CtcDictionary) -> List[_AlignedToken]:
    return [t for t in tokens if t.token_id != dictionary.blank_id()]


def _tokens_to_words(tokens: List[_AlignedToken], dictionary: CtcDictionary) -> List[FrameAlignedWord]:
    words: List[FrameAlignedWord] = []
    accumulated_word = FrameAlignedWord(Text(''), -1, -1)
    for token in tokens:
        str_token = dictionary.decode(TokenizedText([token.token_id]))
        if str_token == '<SIL>':
            continue
        for letter in str_token:
            if letter == ' ':
                if len(accumulated_word.text) > 0:
                    words.append(copy.deepcopy(accumulated_word))
                accumulated_word = FrameAlignedWord(Text(''), -1, -1)
            else:
                if accumulated_word.start_frame == -1:
                    accumulated_word.start_frame = token.start_frame
                accumulated_word.text += letter
                accumulated_word.end_frame = token.end_frame

    if len(accumulated_word.text) > 0:
        words.append(copy.deepcopy(accumulated_word))

    return words


def decode_aligned_tokens(alignment_token_ids: TokenizedText, dictionary: CtcDictionary) -> List[FrameAlignedWord]:
    tokens: List[_AlignedToken] = [_AlignedToken(token_id, i, i + 1)
                                   for i, token_id in enumerate(alignment_token_ids.tokens)]
    tokens = _delete_duplicate_tokens(tokens)
    tokens = _delete_blank_ids(tokens, dictionary)
    words = _tokens_to_words(tokens, dictionary)
    return words
