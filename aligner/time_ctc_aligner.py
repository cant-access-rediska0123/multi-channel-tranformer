import logging
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

from aligner.algo.ctc_alignment import ctc_alignment
from aligner.algo.joint_speakers_ctc_alignment import joint_speakers_alignment
from aligner.aligned_word import FrameAlignedWords, TimeAlignedWord, TimeAlignedWords
from aligner.time_aligner import TimeAligner
from data.text.dictionary.ctc_dictionary import CtcDictionary
from data.text.text import Text

LOG = logging.getLogger()


def _align_sample(i: Tuple[np.ndarray, List[Text]], dictionary: CtcDictionary,
                  use_joint_speakers_alignment: bool, any_speaker_transitions: bool = True) -> FrameAlignedWords:
    ctc_logits, multispeaker_references = i
    speakers_num = len(multispeaker_references)

    if use_joint_speakers_alignment:
        try:
            return joint_speakers_alignment(ctc_logits, multispeaker_references, dictionary,
                                            any_speaker_transitions=any_speaker_transitions)
        except Exception as e:
            LOG.error(f'Failed to use joint speakers alignment with exception: {e}. ' +
                      'Falling back to independent speakers alignment')

    alignments = FrameAlignedWords([])
    for sp in range(speakers_num):
        alignments.append(ctc_alignment(ctc_logits, Text(multispeaker_references[sp].lower()), dictionary))
    return alignments


def _frame_to_time_alignment(speaker_frame_alignments: FrameAlignedWords, ms_in_frame: float) -> TimeAlignedWords:
    alignment = TimeAlignedWords([])
    for speaker_frame_alignment in speaker_frame_alignments:
        time_aligned_speaker_words = []
        for word in speaker_frame_alignment:
            time_aligned_speaker_words.append(
                TimeAlignedWord(word.text, word.start_frame * ms_in_frame, word.end_frame * ms_in_frame))
        alignment.append(time_aligned_speaker_words)
    return alignment


class TimeCtcAligner(TimeAligner):
    def __init__(self, dictionary: CtcDictionary,
                 pool_size: int,
                 ms_in_frame: float,
                 use_joint_speakers_alignment: bool,
                 any_speaker_transitions: bool = False):
        self._dictionary = dictionary
        self._pool = Pool(pool_size)
        self._ms_in_frame = ms_in_frame
        self._use_joint_speakers_alignment = use_joint_speakers_alignment
        self._any_speaker_transitions = any_speaker_transitions

    def __call__(self, references: List[List[Text]], ctc_logits: np.ndarray) -> List[TimeAlignedWords]:
        assert len(ctc_logits.shape) == 3, \
            'Expected 3 dimensions in ctc_logits (batch_size, frames_num, dict_size), got: {}'.format(ctc_logits.shape)
        assert ctc_logits.shape[0] == len(references), \
            'Expected same number of ctc_logits and references, got: {}, {}'.format(
                ctc_logits.shape[0], len(references))

        alignments: List[FrameAlignedWords] = list(self._pool.imap(
            partial(_align_sample,
                    use_joint_speakers_alignment=self._use_joint_speakers_alignment,
                    any_speaker_transitions=self._any_speaker_transitions,
                    dictionary=self._dictionary),
            zip(ctc_logits, references)))

        return list(map(partial(_frame_to_time_alignment, ms_in_frame=self._ms_in_frame), alignments))
