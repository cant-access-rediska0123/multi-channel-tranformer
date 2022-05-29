import random
from typing import List, Tuple, TypeVar

from data.trans import Trans
from factory.factory import make_instance

T = TypeVar('T')


class SampledChainAugmentation(Trans[T, T]):

    def __init__(self,
                 seq: List[Tuple[Trans[T, T], float]]):
        self._seq = seq

    def apply(self, sample: T, **kwargs) -> T:
        for (trans, p) in self._seq:
            if random.Random().random() < p:
                sample = trans(sample, **kwargs)
        return sample

    @staticmethod
    def create(base_class: type, provider) -> 'SampledChainAugmentation[T]':
        seq = []
        for (trans, p) in provider:
            assert 0 <= p <= 1
            seq.append((make_instance(base_class,
                                      trans),
                        p))
        return SampledChainAugmentation(seq)

    @staticmethod
    def create_from_config(base_class: type, provider: List[dict]) -> 'SampledChainAugmentation[T]':
        seq = []
        for config in provider:
            p = config.pop("prob")
            seq.append((config, p))

        return SampledChainAugmentation.create(base_class, seq)
