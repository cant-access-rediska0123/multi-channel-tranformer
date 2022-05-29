from abc import ABC
from collections import deque
from typing import Callable, TypeVar

from data.asr.batch.batch import Batch
from data.asr.batch_builder.batch_builder import BatchBuilder
from data.asr.streaming_dataset.stream_transform import StreamTransform

T = TypeVar('T')


class BlockBatcher(StreamTransform[T, Batch[T]],
                   ABC):

    def __init__(self,
                 batch_size: int,
                 block_size: int,
                 batch_builder: BatchBuilder[T],
                 sort_by: Callable[[T], float] = None):

        self._samples = []
        self._batches = deque()
        self._batch_size = batch_size
        self._block_size = block_size
        self._iter = 0
        self._batch_builder = batch_builder
        self._sort_by = sort_by

    def spit(self) -> Batch[T]:

        if len(self._batches) > 0:
            return self._batches.pop()
        elif len(self._samples) > 0:
            self.__consume_batch_block()
            return self._batches.pop()
        else:
            raise StopIteration

    def charged(self) -> bool:
        return len(self._batches) > 0

    def eat(self, sample: T):
        self._samples.append(sample)
        if len(self._samples) == self._batch_size * self._block_size:
            self.__consume_batch_block()

    def __consume_batch_block(self):
        if self._sort_by:
            samples = sorted(self._samples, key=self._sort_by)
        else:
            samples = self._samples

        for i in range(0, len(samples), self._batch_size):
            self._batches.appendleft(self._batch_builder(samples[i: i + self._batch_size]))

        self._samples.clear()
