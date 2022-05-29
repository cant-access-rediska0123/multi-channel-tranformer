import logging
from typing import List

import numpy as np

from aligner.algo.decode_aligned_tokens import decode_aligned_tokens
from aligner.aligned_word import FrameAlignedWord
from data.text.dictionary.ctc_dictionary import CtcDictionary
from data.text.text import Text, TokenizedText

LOG = logging.getLogger()


def ctc_alignment(ctc_logits: np.ndarray,
                  reference: Text,
                  dictionary: CtcDictionary) -> List[FrameAlignedWord]:
    frames_num = ctc_logits.shape[0]
    tokenized_ref: TokenizedText = dictionary.encode(reference)
    tokens_num = len(tokenized_ref)

    if tokens_num > frames_num:
        LOG.error(f'Reference {reference} too large to align with ctc matrix {ctc_logits.shape}')
        tokenized_ref = TokenizedText(tokenized_ref.tokens[:frames_num])
        tokens_num = frames_num

    dp = np.full((frames_num + 1, tokens_num + 1), fill_value=-np.inf)
    backtrack = np.full((frames_num + 1, tokens_num + 1), fill_value=dictionary.blank_id())
    dp[0][0] = 0
    for f in range(1, frames_num + 1):
        for i in range(tokens_num + 1):
            dp[f][i] = dp[f - 1][i] + ctc_logits[f - 1][dictionary.blank_id()]
            if i > 0:
                token = tokenized_ref.tokens[i - 1]
                token_score = ctc_logits[f - 1][token]
                dp[f][i] = max(dp[f][i], dp[f - 1][i] + token_score)
                if dp[f - 1][i - 1] + token_score > dp[f][i]:
                    dp[f][i] = dp[f - 1][i - 1] + token_score
                    backtrack[f][i] = token
    output = []
    i = tokens_num
    for f in range(frames_num, 0, -1):
        output.append(backtrack[f][i])
        if backtrack[f][i] != dictionary.blank_id():
            i -= 1
    assert i == 0
    output = output[::-1]

    decoded = decode_aligned_tokens(TokenizedText(output), dictionary)

    return decoded
