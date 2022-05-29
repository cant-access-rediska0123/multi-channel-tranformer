from typing import List

from data.asr.batch.property import Property, PropertyCalculator
from data.utterance.utterance import MelUtterance


class AudioDurationCalcer(PropertyCalculator):
    def apply(self, batch: List[MelUtterance], **kwargs) -> Property:
        total = 0
        for sample in batch:
            total += sample.audio.duration_secs
        return Property('total_duration', total)
