import abc
from typing import Generic, Iterator, TypeVar

T = TypeVar('T')
U = TypeVar('U')


class StreamTransform(abc.ABC,
                      Generic[T, U],
                      Iterator[U]):

    @abc.abstractmethod
    def spit(self) -> U:
        pass

    def __next__(self) -> U:
        return self.spit()

    @abc.abstractmethod
    def charged(self) -> bool:
        pass

    @abc.abstractmethod
    def eat(self,
            sample: T):
        pass
