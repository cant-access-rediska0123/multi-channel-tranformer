import os

import numpy as np
import soundfile as sf

from data.asr.datasource.libri_speech.info import LibriSpeechInfo
from data.asr.datasource.source import Source
from data.audio.wave import Wave
from data.text.text import Text
from data.utterance.utterance import RawUtterance


def _read_text(dir_path, file_name: str) -> Text:
    key = file_name.split('.')[0]
    textfile = os.path.join(
        dir_path, '-'.join(file_name.split('-')[:-1]) + '.trans.txt')
    with open(textfile, 'r') as f:
        for line in f.readlines():
            if line.split(' ')[0] == key:
                return Text(' '.join(line.strip().split(' ')[1:]))
    raise Exception(f'No text found for {os.path.join(dir_path, file_name)}')


def _read_audio(audio_path: str) -> Wave:
    data, sample_rate = sf.read(audio_path)
    return Wave(sample_rate, np.array(data))


def parse_sample(dir_path: str, file_name: str) -> RawUtterance:
    wave = _read_audio(os.path.join(dir_path, file_name))
    text = _read_text(dir_path, file_name)
    speaker_id = file_name.split('-')[0]
    return RawUtterance(Text(text), wave, file_name.split('.')[0], speaker_id)


class TrainLibriSpeechSource(Source[RawUtterance]):
    def __init__(self,
                 source: LibriSpeechInfo):
        self._source = source

    def __len__(self):
        return np.inf

    def __iter__(self):
        return self

    def __next__(self):
        cur_pos = np.random.randint(len(self._source))
        dir_path, file_name = self._source.audio_paths[cur_pos]
        return parse_sample(dir_path, file_name)
