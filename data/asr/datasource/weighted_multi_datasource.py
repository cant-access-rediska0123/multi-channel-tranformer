import logging
from typing import Iterator, List, Optional

import numpy as np

from data.asr.datasource.random_data_source import RandomDataSource
from data.asr.datasource.source import Source, T

LOG = logging.getLogger()


class WeightedMultiDataSource(RandomDataSource[T]):
    def __init__(self, datasets: List[Source[T]], weights: List[float]):
        assert len(datasets) == len(weights)
        self._datasets = datasets
        self._iters: List[Optional[Iterator]] = [None for _ in datasets]
        self._weights = weights
        self._weights /= np.sum(weights)

    def __len__(self):
        return np.inf

    def _init_iter(self, reader_idx):
        if self._iters[reader_idx] is not None:
            return
        self._iters[reader_idx] = iter(self._datasets[reader_idx])

    def __next__(self) -> T:
        reader_idx = np.random.choice(len(self._weights), p=self._weights)
        self._init_iter(reader_idx)
        return next(self._iters[reader_idx])
