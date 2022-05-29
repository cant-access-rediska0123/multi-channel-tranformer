import json
import os
from typing import Iterator, List, TypeVar

from data.asr.datasource.libri_speech.info import LibriSpeechInfo
from data.asr.datasource.libri_speech.source import parse_sample
from data.asr.datasource.source import Source
from data.asr.datasource.test_libri_speech.synthetic_gen_parameters import SyntheticGenParams
from data.asr.transforms.multispeaker.utterance_mixer import UtteranceMixer
from data.text.text import MultiSpeakerTexts, Text
from data.utterance.utterance import RawMultiSpeakerUtterance, RawUtterance

T = TypeVar('T')


class TestLibriSpeechReader(Iterator[RawUtterance]):
    def __init__(self, source_info: LibriSpeechInfo):
        self._source_info = source_info
        self._position = 0

    def __next__(self) -> RawUtterance:
        if self._position >= len(self._source_info.audio_paths):
            raise StopIteration
        dir_path, file_name = self._source_info.audio_paths[self._position]
        self._position += 1
        utt = parse_sample(dir_path, file_name)
        return utt


class TestLibriSpeechSource(Source[RawUtterance]):
    def __init__(self, source_info: LibriSpeechInfo):
        self._source_info = source_info

    def __len__(self):
        return len(self._source_info.audio_paths)

    def __iter__(self) -> TestLibriSpeechReader:
        return TestLibriSpeechReader(self._source_info)


class TestSingleSpeakerLibriSpeechReader(Iterator[RawMultiSpeakerUtterance]):
    def __init__(self, source_info: LibriSpeechInfo, total_speakers: int):
        self._total_speakers = total_speakers
        self._reader = TestLibriSpeechReader(source_info)

    def __next__(self) -> RawMultiSpeakerUtterance:
        utt = next(self._reader)
        return RawMultiSpeakerUtterance(
            MultiSpeakerTexts([utt.text] + [Text('') for _ in range(self._total_speakers - 1)]),
            utt.audio, utt.uuid, utt.speaker_id)


class TestSingleSpeakerLibriSpeechSource(Source[RawMultiSpeakerUtterance]):
    def __init__(self, source_info: LibriSpeechInfo, total_speakers: int):
        self._source_info = source_info
        self._total_speakers = total_speakers

    def __len__(self):
        return len(self._source_info.audio_paths)

    def __iter__(self) -> TestSingleSpeakerLibriSpeechReader:
        return TestSingleSpeakerLibriSpeechReader(self._source_info, self._total_speakers)


class TestMultiSpeakerLibriSpeechReader(Iterator[RawMultiSpeakerUtterance]):
    def __init__(self, librispeech_dir: str, sample_pairs: List, params: SyntheticGenParams):
        self._librispeech_dir = librispeech_dir
        self._params = params
        self._sample_pairs = sample_pairs
        self._mixer = UtteranceMixer(speakers_num=params.speakers_num, constant_overlap=self._params.overlap)
        self._position = 0

    def _get_subdir(self, sample) -> str:
        return os.path.join(self._librispeech_dir, sample.split('-')[0], sample.split('-')[1])

    def _get_filename(self, sample) -> str:
        return sample + '.flac'

    def __next__(self) -> RawMultiSpeakerUtterance:
        if self._position >= len(self._sample_pairs):
            raise StopIteration
        sample1, sample2 = self._sample_pairs[self._position]
        self._position += 1
        return self._mixer([parse_sample(self._get_subdir(sample1), self._get_filename(sample1)),
                            parse_sample(self._get_subdir(sample2), self._get_filename(sample2))])


class TestMultiSpeakerLibriSpeechSource(Source[RawMultiSpeakerUtterance]):
    def __init__(self, librispeech_dir: str, sample_pairs_config: str, params: SyntheticGenParams):
        self._librispeech_dir = librispeech_dir
        self._params = params
        with open(sample_pairs_config, 'r') as f:
            self._sample_pairs = json.load(f)

    def __len__(self):
        return len(self._sample_pairs)

    def __iter__(self) -> TestMultiSpeakerLibriSpeechReader:
        return TestMultiSpeakerLibriSpeechReader(self._librispeech_dir, self._sample_pairs, self._params)
