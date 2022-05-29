import logging
import random
from abc import ABC

from data.audio.spectrogram import Spectrogram
from data.trans import Trans

LOG = logging.getLogger()


class SpectrogramAugmetation(Trans[Spectrogram, Spectrogram], ABC):
    pass


class SpecAugment(SpectrogramAugmetation):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779"""

    def apply(self, sample: Spectrogram, **kwargs) -> Spectrogram:
        x = sample.spectrogram
        num_frames = x.shape[0]
        num_features = x.shape[1]

        for _ in range(self._num_frame_regions):
            width = random.randint(1, self._frame_width)
            frame_from = random.randint(0, num_frames - width)
            frame_to = frame_from + width
            # duplicate some feature or just mask it
            val = 0
            if random.uniform(0, 1) > self._zero_prob:
                val = x[random.randint(frame_from, frame_to - 1), random.randint(0, num_features - 1)]
            x[frame_from:frame_to, :] = val

        for _ in range(self._num_feature_regions):
            width = random.randint(0, self._feature_width)
            features_from = random.randint(0, num_features - width)
            features_to = features_from + width

            val = 0
            if random.uniform(0, 1) > self._zero_prob:
                val = x[random.randint(0, num_frames - 1), random.randint(features_from, features_to - 1)]

            x[:, features_from:features_to] = val

        return Spectrogram(x,
                           sample.frame_shift_ms,
                           sample.frame_size_ms)

    def __init__(self,
                 num_frame_regions=1,  # m_T
                 num_feature_regions=1,  # m_F
                 frame_width=10,  # T
                 feature_width=8,  # F
                 zero_prob=0.5):

        self._num_frame_regions = num_frame_regions
        self._num_feature_regions = num_feature_regions

        self._frame_width = frame_width
        self._feature_width = feature_width

        self._zero_prob = zero_prob
