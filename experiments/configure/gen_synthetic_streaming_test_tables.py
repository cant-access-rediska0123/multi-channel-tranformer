import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from data.asr.datasource.libri_speech.source import parse_sample
from data.text.text import Text


@dataclass
class LibriSpeechSample:
    def __init__(self, subdir: str, file: str):
        self.subdir = subdir
        self.file = file
        self.uuid = file.split('.')[0]


def synthetic_2sp_streaming_config_path(librispeech_path: str, test_table: str, overlap: str) -> str:
    return os.path.join(librispeech_path, f'synthetic_2sp_streaming_table_config_{test_table}_{overlap}overlap.json')


def synthetic_1sp_streaming_config_path(librispeech_path: str, test_table: str) -> str:
    return os.path.join(librispeech_path, f'synthetic_1sp_streaming_table_config_{test_table}.json')


def _parse_table(table_path: str) -> Tuple[List[LibriSpeechSample], Dict[str, Text]]:
    audio_files: List[LibriSpeechSample] = []
    references: Dict[str, Text] = {}
    for subdir, _, files in os.walk(table_path):
        for file in files:
            if file.endswith('.flac'):
                audio_files.append(LibriSpeechSample(subdir, file))
            elif file.endswith('.txt'):
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f.readlines():
                        references[line.split(' ')[0]] = Text(' '.join(line.strip().split(' ')[1:]))
    return audio_files, references


class SyntheticLibriSpeechSample:
    def __init__(self, overlap_max_bound: float = 0.0):
        self.samples_to_mix: List[LibriSpeechSample] = []
        self.total_duration = 0.0
        self._prev_duration = None
        self._texts: List[List[Text]] = [[] for _ in range(2)]
        self.overlaps = []
        self._overlap_max_bound = overlap_max_bound

    def add(self, sample: LibriSpeechSample, speaker: int, text: Text):
        cur_duration = parse_sample(sample.subdir, sample.file).audio.duration_secs
        self.total_duration += cur_duration
        if self._prev_duration is not None:
            overlap = np.random.uniform(low=0.0, high=self._overlap_max_bound)
            self.total_duration -= overlap * self._prev_duration
            self.overlaps.append(overlap)
        self._prev_duration = cur_duration
        self.samples_to_mix.append(sample)
        self._texts[speaker].append(text)

    @property
    def texts(self):
        return [' '.join(t) for t in self._texts]


def _join_samples(samples1: List[LibriSpeechSample], samples2: List[LibriSpeechSample],
                  references: Dict[str, Text],
                  overlap_max_bound: float,
                  duration_bounds: Tuple[float, float] = (60 * 1, 60 * 3)) -> Iterable[Tuple[Dict, float]]:
    sample = SyntheticLibriSpeechSample(overlap_max_bound)
    want_duration = np.random.uniform(low=duration_bounds[0], high=duration_bounds[1])
    for samples in zip(samples1, samples2):
        for speaker, s in enumerate(samples):
            sample.add(s, speaker=speaker, text=references[s.uuid])
            if sample.total_duration >= want_duration:
                yield {
                          'samples_to_mix': [s.file for s in sample.samples_to_mix],
                          'reference': sample.texts,
                          'overlaps': sample.overlaps,
                      }, sample.total_duration
                sample = SyntheticLibriSpeechSample(overlap_max_bound)


def _merge_samples(samples: List[LibriSpeechSample], references: Dict[str, Text],
                   duration_bounds: Tuple[float, float] = (60 * 1, 60 * 3)) -> Iterable[Tuple[Dict, float]]:
    sample = SyntheticLibriSpeechSample()
    want_duration = np.random.uniform(low=duration_bounds[0], high=duration_bounds[1])
    for s in samples:
        sample.add(s, speaker=0, text=references[s.uuid])
        if sample.total_duration >= want_duration:
            yield {
                      'samples_to_mix': [s.file for s in sample.samples_to_mix],
                      'reference': sample.texts,
                      'overlaps': sample.overlaps,
                  }, sample.total_duration
            sample = SyntheticLibriSpeechSample()


def _gen_synthetic_2sp_samples(
        audio_files: List[LibriSpeechSample],
        recognitions: Dict[str, Text],
        overlap_max_bound: float,
        single_duration_min_bound: float = 10.0) -> Tuple[List, List[float]]:
    np.random.seed(228)
    speakers_durations: Dict[str, float] = defaultdict(float)
    speakers_samples: Dict[str, List[LibriSpeechSample]] = defaultdict(list)
    for sample in audio_files:
        utt = parse_sample(sample.subdir, sample.file)
        if utt.audio.duration_secs < single_duration_min_bound:
            continue
        speakers_durations[utt.speaker_id] += utt.audio.duration_secs
        speakers_samples[utt.speaker_id].append(sample)
    sorted_speakers_durations = sorted(speakers_durations.items(),
                                       key=lambda x: (x[1], x[0]))
    config, durations = [], []
    for i in range(0, len(sorted_speakers_durations) - 1, 2):
        speaker1 = sorted_speakers_durations[i][0]
        speaker2 = sorted_speakers_durations[i + 1][0]
        for sample, duration in _join_samples(
                speakers_samples[speaker1], speakers_samples[speaker2], recognitions, overlap_max_bound):
            config.append(sample)
            durations.append(duration)
    return config, durations


def _gen_synthetic_1sp_samples(audio_files: List[LibriSpeechSample],
                               recognitions: Dict[str, Text]) -> Tuple[List, List[float]]:
    np.random.seed(228)
    speakers_durations: Dict[str, float] = defaultdict(float)
    speakers_samples: Dict[str, List[LibriSpeechSample]] = defaultdict(list)
    for sample in audio_files:
        utt = parse_sample(sample.subdir, sample.file)
        speakers_durations[utt.speaker_id] += utt.audio.duration_secs
        speakers_samples[utt.speaker_id].append(sample)
    sorted_speakers_samples = sorted(speakers_samples.items(), key=lambda x: (x[0]))
    config, durations = [], []
    for i in range(len(sorted_speakers_samples)):
        samples = sorted_speakers_samples[i][1]
        for sample, duration in _merge_samples(samples, recognitions):
            config.append(sample)
            durations.append(duration)
    return config, durations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--librispeech_path', type=str, default='LibriSpeech')
    parser.add_argument('--test_tables', type=str, nargs='*', default=['test-clean', 'test-other'])

    overlap_max_bound = 0.1

    parsed_args = parser.parse_args()

    librispeech_path = parsed_args.librispeech_path

    for test_table in parsed_args.test_tables:
        audio_files, recognitions = _parse_table(os.path.join(librispeech_path, test_table))
        config, durations = _gen_synthetic_2sp_samples(audio_files, recognitions,
                                                       overlap_max_bound=overlap_max_bound)

        no_overlap_config = [{
            'samples_to_mix': s['samples_to_mix'],
            'reference': s['reference'],
            'overlaps': [0.0 for _ in range(s['overlaps'])],
        } for s in config]

        for cfg, name in [(config, '0_10'), (no_overlap_config, '0')]:
            synthetic_config_path = synthetic_2sp_streaming_config_path(librispeech_path, test_table, name)
            if os.path.exists(synthetic_config_path):
                print(f'{synthetic_config_path}: already configured')
            else:
                print(f'{test_table}: configured {len(cfg)} two-speaker streaming test samples '
                      f'with {name} overlaps out of {len(audio_files)} single-speaker test samples')
                print(
                    f' Avg duration: {round(np.mean(durations).item())}s\n'
                    f' Max duration: {round(max(durations))}s\n'
                    f' Min_duration: {round(min(durations))}s')
                with open(synthetic_config_path, 'w') as f:
                    json.dump(cfg, f)

        synthetic_config_path = synthetic_1sp_streaming_config_path(librispeech_path, test_table)
        if os.path.exists(synthetic_config_path):
            print(f'{synthetic_config_path}: already configured')
        else:
            config, durations = _gen_synthetic_1sp_samples(audio_files, recognitions)
            print(f'{test_table}: configured {len(config)} single-speaker streaming test samples '
                  f'out of {len(audio_files)} single-speaker test samples')
            print(
                f' Avg duration: {round(np.mean(durations).item())}s\n'
                f' Max duration: {round(max(durations))}s\n'
                f' Min duration: {round(min(durations))}s')
            with open(synthetic_config_path, 'w') as f:
                json.dump(config, f)
