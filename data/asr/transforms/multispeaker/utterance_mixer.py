from typing import List, Optional

import numpy as np

from data.audio.wave import Wave
from data.text.text import MultiSpeakerTexts, Text
from data.trans import Trans
from data.utterance.utterance import RawMultiSpeakerUtterance, RawUtterance


class UtteranceMixer(Trans[List[RawUtterance], RawMultiSpeakerUtterance]):
    def __init__(self,
                 speakers_num: int,
                 constant_overlap: Optional = None,
                 # only if constant_gap is None:
                 min_offset_sec=1,
                 min_relative_offset: float = 0.3,
                 max_shift_sec: float = 8,
                 min_shift_sec: float = 0.5,
                 ):
        self._speakers_num = speakers_num
        self._constant_overlap = constant_overlap
        self._min_offset_sec = min_offset_sec
        self._min_relative_offset = min_relative_offset
        self._max_shift_sec = max_shift_sec
        self._min_shift_sec = min_shift_sec

    def mix_audios(self, audios: List[Wave]) -> Wave:

        rate = audios[0].sample_rate
        assert all(a.sample_rate == rate for a in audios)

        if self._constant_overlap is None:
            offsets = []
            for s1, s2 in zip(audios[:-1], audios[1:]):
                low = min(max(self._min_offset_sec / s1.duration_secs,
                              self._min_relative_offset) * len(s1.data), len(s1.data) - 1)
                high = len(s1.data)
                offsets.append(np.random.randint(int(low), int(high)))
        elif isinstance(self._constant_overlap, list):
            assert len(self._constant_overlap) + 1 == len(audios)
            offsets = [int((1.0 - overlap) * len(w.data)) for w, overlap in
                       zip(audios[:-1], self._constant_overlap)]
        else:
            offsets = [int((1.0 - self._constant_overlap) * len(w.data)) for w in audios[:-1]]
        silence_offset_lengths = np.cumsum([0] + offsets)

        # Add silence offset before asr
        audios = [np.concatenate((np.zeros(offset_len), a.data)) if offset_len > 0 else a.data[-offset_len:]
                  for offset_len, a in zip(silence_offset_lengths, audios)]
        total_audio_len = max(len(w) for w in audios)

        # Add silence offset after asr
        audios = np.array([np.concatenate((w, np.zeros(total_audio_len - len(w)))) for w in audios]).astype(float)

        audio = audios.sum(axis=0) / self._speakers_num

        return Wave(rate, audio)

    def mix_texts(self, texts: List[Text]) -> MultiSpeakerTexts:
        return MultiSpeakerTexts(texts + [Text('') for _ in range(self._speakers_num - len(texts))])

    def apply(self, samples: List[RawUtterance], **kwargs) -> RawMultiSpeakerUtterance:
        mixed_wave = self.mix_audios([s.audio for s in samples])
        mixed_texts = self.mix_texts([s.text for s in samples])
        utt = RawMultiSpeakerUtterance(mixed_texts,
                                       mixed_wave,
                                       '_'.join(s.uuid for s in samples),
                                       '_'.join(s.speaker_id for s in samples))
        return utt
