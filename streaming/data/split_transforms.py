import logging
from typing import List

from data.audio.spectrogram import Spectrogram
from data.text.text import MultiSpeakerTexts, MultiSpeakerTokenizedTexts
from data.text.text_processor.text_processor import TextProcessor
from data.trans import Trans
from data.utterance.utterance import MelMultiSpeakerUtterance, RawMultiSpeakerUtterance
from streaming.data.bits import BitInfo, MelBitUtterance, SampleInfo, SplitMultiSpeakerMelUtterance

LOG = logging.getLogger()


class SplitTransforms(Trans[RawMultiSpeakerUtterance, SplitMultiSpeakerMelUtterance]):
    def __init__(self,
                 transforms: Trans[RawMultiSpeakerUtterance, MelMultiSpeakerUtterance],
                 text_processor: TextProcessor,
                 window_size_ms: int, window_shift_ms: int):
        self._transforms = transforms
        self._window_size_ms = window_size_ms
        self._window_shift_ms = window_shift_ms
        self._text_processor = text_processor

    def apply(self, sample: RawMultiSpeakerUtterance, **kwargs) -> SplitMultiSpeakerMelUtterance:
        mel_bits: List[MelBitUtterance] = []

        audio_duration_ms = sample.audio.duration_secs * 1000
        window_start_ms = 0
        mel: MelMultiSpeakerUtterance = self._transforms.apply(
            RawMultiSpeakerUtterance(MultiSpeakerTexts([]), sample.audio, sample.uuid, sample.speaker_id))

        while True:
            window_end_ms = min(audio_duration_ms, window_start_ms + self._window_size_ms)

            window_start_frames = int(window_start_ms / mel.audio.frame_shift_ms)
            window_end_frames = int(
                (window_end_ms - mel.audio.frame_size_ms + mel.audio.frame_shift_ms) / mel.audio.frame_shift_ms)

            bit_spec = mel.audio.spectrogram[window_start_frames:window_end_frames]

            mel_bit = MelMultiSpeakerUtterance(
                MultiSpeakerTokenizedTexts([]),
                Spectrogram(bit_spec, mel.audio.frame_shift_ms, mel.audio.frame_size_ms),
                sample.uuid, sample.speaker_id)

            mel_bits.append(MelBitUtterance(mel_bit.audio, BitInfo(
                window_start_ms, window_end_ms,
                SampleInfo(audio_duration_ms, sample.uuid,
                           [self._text_processor.process_text(t) for t in sample.text]))))
            if window_end_ms >= audio_duration_ms:
                break
            window_start_ms += self._window_shift_ms

        return SplitMultiSpeakerMelUtterance(sample.text, mel_bits, sample.uuid, sample.speaker_id)
