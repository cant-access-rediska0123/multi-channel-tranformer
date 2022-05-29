import abc
from typing import Generic

from torch.utils.data.dataset import IterableDataset

from data.asr.datasource.source import Source, T


class RandomDataSource(Generic[T], abc.ABC, IterableDataset, Source[T]):
    @abc.abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self

    def __getitem__(self, item):
        return next(self)

    @abc.abstractmethod
    def __len__(self):
        pass
