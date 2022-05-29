from typing import Callable

from data.asr.batch_builder.batch_builder import BatchBuilder
from data.asr.streaming_dataset.block_batcher import BlockBatcher
from data.asr.streaming_dataset.stream_transform import StreamTransform
from streaming.data.bits import SplitMultiSpeakerMelUtterance
from streaming.data.bits_batch import BitsMultiSpeakerBatch, MelBitUtterance


class BitsBatcher(StreamTransform[SplitMultiSpeakerMelUtterance, BitsMultiSpeakerBatch]):

    def __init__(self, batch_size: int,
                 block_size: int,
                 batch_builder: BatchBuilder[BitsMultiSpeakerBatch],
                 sort_by: Callable[[MelBitUtterance], float] = None):
        self._batcher = BlockBatcher[MelBitUtterance](batch_size, block_size, batch_builder, sort_by)

    def spit(self) -> BitsMultiSpeakerBatch:
        return self._batcher.spit()

    def charged(self) -> bool:
        return self._batcher.charged()

    def eat(self, sample: SplitMultiSpeakerMelUtterance):
        for bit in sample.audio:
            self._batcher.eat(bit)
