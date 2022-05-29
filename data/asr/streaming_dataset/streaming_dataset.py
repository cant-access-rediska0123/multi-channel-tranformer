import logging
from typing import Optional, TypeVar

import numpy as np

from data.asr.datasource.random_data_source import RandomDataSource
from data.asr.datasource.source import Reader, Source
from data.asr.streaming_dataset.stream_transform import StreamTransform
from data.trans import Trans
from distributed.utils import before_new_sample
from factory.factory import make_instance

LOG = logging.getLogger()

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class StreamingDataset(RandomDataSource[V]):
    def __init__(self,
                 transforms: Trans[T, U],
                 batcher: StreamTransform[U, V],
                 source: Source[T],
                 size: Optional[int] = None,
                 **kwargs):
        self._batcher: StreamTransform[U, V] = batcher
        self._source: Source[T] = make_instance(Source, source)
        self._extractor: Trans[T, U] = make_instance(Trans, transforms)
        self._reader: Optional[Reader[T]] = None
        self._size = size if size is not None else np.inf

    def __len__(self):
        return self._size

    def __next__(self) -> V:
        before_new_sample()
        if self._reader is None:
            self._reader = iter(self._source)
        transform = make_instance(StreamTransform, self._batcher)
        while not transform.charged():
            try:
                raw_sample: T = next(self._reader)
            except StopIteration:
                return transform.spit()

            try:
                sample: U = self._extractor(raw_sample)
            except Exception as e:
                LOG.warning(f"Can't parse sample with exception {repr(e)}")
                continue

            assert sample is not None

            try:
                transform.eat(sample)
            except Exception as e:
                LOG.warning(f"Eat exception, can't consume sample with exception {repr(e)}")

        return transform.spit()
