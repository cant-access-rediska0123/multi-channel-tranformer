import abc
import logging
from typing import Callable, List, Optional

import numpy as np

from aligner.algo.decode_aligned_tokens import decode_aligned_tokens
from aligner.aligned_word import FrameAlignedWords
from data.text.dictionary.ctc_dictionary import CtcDictionary
from data.text.text import Text, TokenizedText

LOG = logging.getLogger()


class CtcPrior:
    @abc.abstractmethod
    def __call__(self, ctc_logits: np.ndarray) -> np.ndarray:
        pass


class FunctionCtcPrior(CtcPrior):
    def __init__(self, function: Callable[[float], float], prior_weight: float):
        self._f = function
        self._w = prior_weight

    def __call__(self, ctc_logits: np.ndarray):
        spec_len = ctc_logits.shape[0]
        dict_len = ctc_logits.shape[1]
        prior_mat = np.array([[self._f(i / spec_len) + 1e-05] for i in range(spec_len)])
        prior_mat /= prior_mat.sum()
        prior_mat = np.log(prior_mat)
        prior_mat = np.repeat(prior_mat, dict_len, axis=1)
        return ctc_logits * (1.0 - self._w) + prior_mat * self._w


class NoPrior(CtcPrior):
    def __call__(self, ctc_logits: np.ndarray):
        return ctc_logits


class LinearCtcPrior(FunctionCtcPrior):
    def __init__(self, k: float, b: float, prior_weight: float):
        super().__init__(lambda x: k * x + b, prior_weight)


# Aligns 2 speakers references to single ctc matrix independently
def joint_speakers_alignment(ctc_logits: np.ndarray,
                             references: List[Text],
                             dictionary: CtcDictionary,
                             speaker_priors: Optional[List[CtcPrior]] = None,
                             # speaker_priors: Optional[List[CtcPrior]] = [
                             #     FunctionCtcPrior(lambda x: np.sqrt(1 - x), prior_weight=0.1),
                             #     FunctionCtcPrior(lambda x: np.sqrt(x), prior_weight=0.1),
                             # ],
                             any_speaker_transitions: bool = False) -> FrameAlignedWords:
    sp_num = len(references)
    if sp_num != 2:  # TODO(rediska)
        raise Exception(f'Only 2 speakers joint alignment is supported, got: {sp_num} speakers')

    if speaker_priors is None:
        speaker_priors = [NoPrior() for _ in range(sp_num)]

    references = [Text(h.lower()) for h in references]
    frames_num = ctc_logits.shape[0]
    tokenized_refs: List[List[int]] = [dictionary.encode(ref).tokens for ref in references]

    if sum(len(t) for t in tokenized_refs) > frames_num:
        raise Exception(f'References {references} too large for joint speakers alignment')

    len1, len2 = len(tokenized_refs[0]), len(tokenized_refs[1])
    max_joint_logit_dp = np.full((frames_num + 1, len1 + 1, len2 + 1, sp_num), fill_value=-np.inf)
    dp_backtrack = np.full((frames_num + 1, len1 + 1, len2 + 1, sp_num, 3), fill_value=None)
    skip_token_position = np.full((frames_num + 1, len1 + 1, len2 + 1, sp_num), fill_value=False)

    max_joint_logit_dp[:, 0, 0, :] = 0
    dp_backtrack[:, 0, 0, :, :] = 0
    skip_token_position[:, 0, 0, :] = True

    speaker_ctc_logits = [speaker_prior(ctc_logits) for speaker_prior in speaker_priors]

    for frame in range(1, frames_num + 1):
        for pref1 in range(len1 + 1):
            for pref2 in range(len2 + 1):
                for cur_speaker in range(sp_num):
                    if cur_speaker == 0 and pref1 == 0:
                        continue
                    if cur_speaker == 1 and pref2 == 0:
                        continue

                    cur_token = tokenized_refs[cur_speaker][(pref1 if cur_speaker == 0 else pref2) - 1]
                    cur_logit = speaker_ctc_logits[cur_speaker][frame - 1, cur_token]
                    cur_logit_if_blank = speaker_ctc_logits[cur_speaker][frame - 1, dictionary.blank_id()]

                    if any_speaker_transitions:
                        prev_speaker_cases = range(sp_num)
                    else:
                        if (cur_speaker == 0 and pref2 in [len2, 0]) or (
                                cur_speaker == 1 and pref1 in [len1, 0]) or cur_token == dictionary.sil_id():
                            prev_speaker_cases = range(sp_num)
                        else:
                            prev_speaker_cases = [cur_speaker]

                    for prev_speaker in prev_speaker_cases:
                        blank_case_logit = max_joint_logit_dp[frame - 1, pref1, pref2, prev_speaker] + max(
                            cur_logit_if_blank, cur_logit)
                        backtrack = [pref1, pref2, prev_speaker]
                        if max_joint_logit_dp[frame, pref1, pref2, cur_speaker] < blank_case_logit:
                            max_joint_logit_dp[frame, pref1, pref2, cur_speaker] = blank_case_logit
                            dp_backtrack[frame, pref1, pref2, cur_speaker] = backtrack
                            skip_token_position[frame, pref1, pref2, cur_speaker] = True

                        if cur_speaker == 0:
                            token_case_logit = max_joint_logit_dp[
                                                   frame - 1, pref1 - 1, pref2, prev_speaker] + cur_logit \
                                if pref1 > 0 else None
                            backtrack = [pref1 - 1, pref2, prev_speaker]
                        else:
                            token_case_logit = max_joint_logit_dp[
                                                   frame - 1, pref1, pref2 - 1, prev_speaker] + cur_logit \
                                if pref2 > 0 else None
                            backtrack = [pref1, pref2 - 1, prev_speaker]
                        if token_case_logit is not None and token_case_logit > max_joint_logit_dp[
                            frame, pref1, pref2, cur_speaker]:
                            max_joint_logit_dp[frame, pref1, pref2, cur_speaker] = token_case_logit
                            dp_backtrack[frame, pref1, pref2, cur_speaker] = backtrack
                            skip_token_position[frame, pref1, pref2, cur_speaker] = False

    aligned_tokens = np.full((sp_num, frames_num), fill_value=dictionary.blank_id())
    pref1, pref2 = len1, len2
    cur_speaker = max_joint_logit_dp[frames_num, pref1, pref2].argmax()

    assert max_joint_logit_dp[frames_num, pref1, pref2, cur_speaker] != -np.inf

    for frame in range(frames_num, 0, -1):
        if not skip_token_position[frame, pref1, pref2, cur_speaker]:
            cur_token = tokenized_refs[cur_speaker][(pref1 if cur_speaker == 0 else pref2) - 1]
            aligned_tokens[cur_speaker, frame - 1] = cur_token

        pref1, pref2, cur_speaker = dp_backtrack[frame, pref1, pref2, cur_speaker]

    assert pref1 == 0 and pref2 == 0

    return FrameAlignedWords([decode_aligned_tokens(TokenizedText(aligned_tokens[sp]), dictionary)
                              for sp in range(sp_num)])
