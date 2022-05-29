from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

from data.asr.batch.batch import Batch
from data.trans import Trans

V = TypeVar('V')


@dataclass
class Property(Generic[V]):
    name: str
    value: V


class PropertyCalculator(Trans[Batch, Property], ABC):
    pass
