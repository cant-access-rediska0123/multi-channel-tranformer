import logging
from typing import List

import numpy as np

from data.asr.datasource.random_data_source import RandomDataSource
from data.asr.datasource.source import Source
from data.utterance.utterance import RawUtterance

LOG = logging.getLogger()


class MultiSpeakerSource(RandomDataSource[List[RawUtterance]]):
    def __init__(self, single_speaker_source: Source[RawUtterance], speakers_num: int):
        self._single_speaker_source = single_speaker_source
        self._speakers_num = speakers_num
        self._iter = None

    def __len__(self):
        return np.inf

    def _init_iter(self):
        if self._iter is not None:
            return
        self._iter = iter(self._single_speaker_source)

    def _new_utterance(self, speaker_ids: List[str]) -> RawUtterance:
        new_utt = next(self._iter)
        while new_utt.speaker_id in speaker_ids:
            new_utt = next(self._iter)
        return new_utt

    def __next__(self) -> List[RawUtterance]:
        self._init_iter()
        utts: List[RawUtterance] = []
        speaker_ids = []
        for speaker in range(self._speakers_num):
            new_utt = self._new_utterance(speaker_ids)
            speaker_ids.append(new_utt.speaker_id)
            utts.append(new_utt)
        return utts
