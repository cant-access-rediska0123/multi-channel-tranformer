from dataclasses import dataclass
from typing import Optional


@dataclass
class SyntheticGenParams:
    speakers_num: int
    overlap: Optional[float]

    def __str__(self):
        if self.overlap is None:
            return f'{self.speakers_num}sp'
        else:
            return f'{self.speakers_num}sp_{self.overlap}overlap'
