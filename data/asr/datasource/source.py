import abc
from typing import Iterable, TypeVar

T = TypeVar('T')


class Reader(Iterable[T]):

    @abc.abstractmethod
    def __next__(self) -> T:
        pass


class Source(Iterable[T]):

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __iter__(self) -> Reader[T]:
        pass
