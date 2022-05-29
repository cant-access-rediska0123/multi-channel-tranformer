import os
from typing import List, Tuple

from data.asr.datasource.source_info import SourceInfo


class LibriSpeechInfo(SourceInfo):
    def __init__(self, path: str):
        super(LibriSpeechInfo, self).__init__(path)
        self._audios = [(subdir, file)
                        for subdir, _, files in os.walk(self._path)
                        for file in files if file.endswith('.flac')]
        self._size = len(self._audios)

    def __len__(self) -> int:
        return self._size

    @property
    def audio_paths(self) -> List[Tuple[str, str]]:
        return self._audios
