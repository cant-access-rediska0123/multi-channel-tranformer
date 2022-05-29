import argparse
import json
import os
from typing import List, Tuple

import numpy as np


def synthetic_config_path(librispeech_path: str, test_table: str) -> str:
    return os.path.join(librispeech_path, f'synthetic_table_config_{test_table}.json')


def _get_audio_files(table_path: str) -> List[str]:
    return [file.split('.')[0]
            for subdir, _, files in os.walk(table_path)
            for file in files if file.endswith('.flac')]


def _configure_ids_to_mix(audio_files: List[str]) -> List[Tuple[str, str]]:
    audio_files.sort()
    np.random.seed(228)
    np.random.shuffle(audio_files)
    pairs = []
    for i in range(0, len(audio_files) - 1, 2):
        sample1 = audio_files[i]
        sample2 = audio_files[i + 1]
        speaker_id1 = sample1.split('-')[0]
        speaker_id2 = sample2.split('-')[0]
        if speaker_id1 == speaker_id2:
            continue
        pairs.append((speaker_id1, speaker_id2))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--librispeech_path', type=str, default='LibriSpeech')
    parser.add_argument('--test_tables', type=str, nargs='*', default=['test-clean', 'test-other'])

    parsed_args = parser.parse_args()

    librispeech_path = parsed_args.librispeech_path

    for test_table in parsed_args.test_tables:
        path = synthetic_config_path(librispeech_path, test_table)
        if os.path.exists(path):
            print(f'{test_table}: already configured')
            continue
        audio_files = _get_audio_files(os.path.join(librispeech_path, test_table))
        synthetic_ids = _configure_ids_to_mix(audio_files)
        print(f'{test_table}: configured {len(synthetic_ids)} two-speaker test samples '
              f'out of {len(audio_files)} single-speaker test samples')
        with open(path, 'w') as f:
            json.dump(synthetic_ids, f)
