import abc
from typing import Generic, List, Optional, TypeVar

from data.asr.batch.batch import Batch
from data.asr.batch.property import Property
from data.trans import Trans

T = TypeVar('T')


class BatchBuilder(Generic[T], abc.ABC):

    def __init__(self, properties_setters: Optional[List[Trans[Batch[T], Property]]] = None):
        if properties_setters is None:
            self._properties_setters = []
        else:
            self._properties_setters = properties_setters

    @abc.abstractmethod
    def _build(self, batch: List[T]) -> Batch[T]:
        pass

    def __call__(self, batch_src: List[T]) -> Batch[T]:
        batch: Batch[T] = self._build(batch_src)
        for setter in self._properties_setters:
            prop = setter(batch_src)
            batch.properties[prop.name] = prop.value
        return batch
