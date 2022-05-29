import abc
from typing import Generic, TypeVar

T = TypeVar('T')


class Batch(abc.ABC, Generic[T]):
    def __init__(self):
        self.properties = {}
