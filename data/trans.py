import abc
from abc import ABC
from typing import Generic, List, TypeVar

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class Trans(Generic[T, U],
            ABC):

    @abc.abstractmethod
    def apply(self, sample: T, **kwargs) -> U:
        pass

    def __call__(self, sample: T, **kwargs) -> U:
        return self.apply(sample, **kwargs)


class ChainCall(Trans[T, T]):

    def __init__(self, chain: List[Trans[T, T]]):
        self._seq = chain

    def apply(self, sample: T, **kwargs) -> T:
        for trans in self._seq:
            sample = trans(sample, **kwargs)
        return sample
