import argparse
import json
import logging
import os
from typing import Dict

import numpy as np
from scipy.io import wavfile

from data.asr.datasource.source import Source
from data.audio.wave import Wave
from data.text.dictionary import Dictionary
from data.text.text_processor import TextProcessor
from data.trans import Trans
from data.utterance.utterance import MelMultiSpeakerUtterance, MultiSpeakerTexts, MultiSpeakerTokenizedTexts, \
    RawMultiSpeakerUtterance, Spectrogram, Text
from experiments.train.train import configure_model as configure_diarization_transformer, \
    parse_train_config
from experiments.train.transformer_data_module import TransformerDataModule
from factory.factory import make_instance
from models.generator.beam_generator import BeamDiarizationGenerator
from streaming.graph.streaming_graph_parameters import StreamingGraphParameters
from streaming.streaming_evaluation import StreamingEvaluator

logging.basicConfig(filename='logs/log.txt', level=logging.INFO)


class DiskSource(Source[RawMultiSpeakerUtterance]):
    def __init__(self, path: str):
        if os.path.isdir(path):
            self._wav_paths = [os.path.join(subdir, file)
                               for subdir, _, files in os.walk(path)
                               for file in files if file.endswith('.wav')]
        elif path.endswith('.wav'):
            self._wav_paths = [path]
        else:
            raise Exception(f'Unknown audio source: {path}')

    def __len__(self):
        return len(self._wav_paths)

    def _parse_sample(self, wav_path: str) -> RawMultiSpeakerUtterance:
        sample_rate, audio = wavfile.read(wav_path)
        audio = np.array(audio)
        min = np.iinfo(audio.dtype).min
        max = np.iinfo(audio.dtype).max
        audio = (audio.astype(np.float) * 2 - max - min) / (max - min) * np.iinfo(np.int16).max
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        texts = MultiSpeakerTexts([Text('') for _ in range(2)])
        sample = RawMultiSpeakerUtterance(texts, Wave(sample_rate, audio), wav_path)
        return sample

    def __iter__(self):
        return iter(self._parse_sample(wav_path) for wav_path in self._wav_paths)


class NoSpectrogramCalculation(Trans[RawMultiSpeakerUtterance, MelMultiSpeakerUtterance]):
    def apply(self, sample: RawMultiSpeakerUtterance, **kwargs) -> MelMultiSpeakerUtterance:
        spec = Spectrogram(sample.audio.data[..., np.newaxis],
                           1000 / sample.audio.sample_rate, 1000 / sample.audio.sample_rate,
                           orig_audio=sample.audio)
        return MelMultiSpeakerUtterance(MultiSpeakerTokenizedTexts([]), spec, sample.uuid, sample.speaker_id)


class StreamingDiarizationTransformerInference:
    def __init__(self,
                 transformer_checkpoint_path: str, transformer_train_config: Dict, beam_size: int,
                 jasper_dictionary_path: str,
                 streaming_parameters_config: Dict,
                 use_joint_speakers_alignment: bool,
                 batch_size: int = 1,
                 pool_size: int = 1,
                 max_streaming_graphs_parallel: int = 1,
                 device: str = 'cuda:0',
                 ):
        from models.nemo_jasper import NemoJasper
        from streaming.recognition_model.transformer_recognition_model import TransformerRecognitionModel
        self._params = StreamingGraphParameters(**streaming_parameters_config)
        self._transformer = configure_diarization_transformer(
            transformer_checkpoint_path, transformer_train_config)
        self._transformer.to(device)
        self._transformer.eval()
        self._jasper = NemoJasper(device)
        self._jasper_dictionary = make_instance(Dictionary, jasper_dictionary_path)
        self._use_joint_speakers_alignment = use_joint_speakers_alignment

        self._text_processor: TextProcessor = make_instance(Trans, transformer_train_config['text_processor_config'])

        self._diarization_transformer_transforms = TransformerDataModule(
            transformer_train_config).construct_transforms()
        self._jasper_transforms = NoSpectrogramCalculation()
        self._alignment_model = TransformerRecognitionModel(
            text_generator=BeamDiarizationGenerator(
                self._transformer.encoder, self._transformer.decoder, self._transformer.speakers_num,
                self._transformer.text_decoder, beam_size=beam_size),
            jasper=self._jasper,
            ctc_dictionary=self._jasper_dictionary,
            ms_in_frame=20,
            pool_size=pool_size,
            use_joint_speakers_alignment=use_joint_speakers_alignment,
        )
        self._batch_size = batch_size
        self._pool_size = pool_size
        self._max_streaming_graphs_parallel = max_streaming_graphs_parallel
        self._streaming_args = {
            "models_transforms": [self._diarization_transformer_transforms, self._jasper_transforms],
            "text_processor": self._text_processor,
            "model": self._alignment_model,
            "params": self._params,
            "batch_size": self._batch_size,
            "max_streaming_graphs_parallel": self._max_streaming_graphs_parallel,
            "pool_size": self._pool_size,
        }

    def __call__(self, source: Source[RawMultiSpeakerUtterance]) -> Dict:
        streaming = StreamingEvaluator(data_source=source, **self._streaming_args)
        recognitions = {}
        for sample_info, sample_hyp in streaming.evaluate():
            recognitions[sample_info.orig_uuid] = sample_hyp
        return recognitions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # checkpoints
    parser.add_argument('--transformer_train_config_path', type=str,
                        default='experiments/configs/train_configs/diarization_transformer_parallel.json')
    parser.add_argument('--transformer_checkpoint_path', type=str, required=True,
                        help='Path to Transformer recognition model checkpoint')
    parser.add_argument('--beam_size', type=int, default=5)

    parser.add_argument('--jasper_dictionary_config_path', type=str,
                        default='experiments/configs/dictionaries/ctc_eng.json')

    # data
    parser.add_argument('--wav_data_path', type=str, required=True,
                        help='Path to the directory with .wav audio files to recognize')

    # evaluation
    parser.add_argument('--streaming_parameters_config_path', type=str,
                        default='experiments/configs/streaming/streaming_parameters.json')
    parser.add_argument('--use_joint_speakers_alignment', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--pool_size', type=int, default=8)
    parser.add_argument('--max_processed_samples', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')

    args = parser.parse_args()

    transformer_config = parse_train_config(args.transformer_train_config_path)
    jasper_config = parse_train_config(args.jasper_train_config_path)
    with open(args.streaming_parameters_config_path, 'r') as f:
        streaming_params_config = json.load(f)

    streaming_inference = StreamingDiarizationTransformerInference(
        args.transformer_checkpoint_path, transformer_config, args.beam_size,
        args.jasper_dictionary_config_path,
        streaming_params_config, args.use_joint_speakers_alignment,
        args.batch_size, args.pool_size, args.max_processed_samples, args.device)

    source = DiskSource(args.wav_data_path)
    recognitions = streaming_inference(source)
    with open(args.output_path, 'r') as f:
        json.dump(recognitions, f)
