import argparse
import json
import tempfile
from typing import Dict

import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
from speechbrain.pretrained import SepformerSeparation as separator
from tqdm import tqdm

from data.asr.batch.speech_batch import SpeechBatch
from data.asr.batch_builder.multispeaker_speech_batch_builder import MultiSpeakerSpeechBatchBuilder
from data.asr.transforms.mel_builder.multispeaker_mel_builder import MelMultiSpeakerUtteranceBuilder
from data.asr.transforms.spectrogram_builder.spectrogram_builder import SpectrogramBuilder
from data.audio.wave import Wave
from data.text.dictionary.sentencepiece_dictionary import Dictionary, SentencePieceDictionary
from data.text.text import MultiSpeakerTexts
from data.text.text_processor.multispeaker_text_processor import MultiSpeakerTextProcessor
from data.trans import ChainCall, Trans
from data.utterance.utterance import MelMultiSpeakerUtterance, RawMultiSpeakerUtterance
from experiments.train.train import configure_model, parse_train_config
from experiments.train.transformer_data_module import TransformerDataModule
from factory.factory import make_instance
from models.generator.beam_generator import BeamDiarizationGenerator
from torch_modules.metrics.speaker_independent_wer import SpeakerIndependentWer


def _resample(data: np.array, sample_rate: int, new_sample_rate: int) -> np.array:
    number_of_samples = round(len(data) * float(new_sample_rate) / sample_rate)
    return sps.resample(data, number_of_samples)


class SepformerInference:
    def __init__(self, transformer_checkpoint_path: str, transformer_train_config: Dict,
                 beam_size: int, device: str):
        self._sepformer = separator.from_hparams(
            source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')
        transformer = configure_model(transformer_checkpoint_path, transformer_train_config)
        transformer.to(device)
        self._generator = BeamDiarizationGenerator(
            transformer.encoder, transformer.decoder,
            transformer.speakers_num, transformer.text_decoder, beam_size=beam_size)
        dictionary: SentencePieceDictionary = make_instance(Dictionary, transformer_train_config['dictionary_config'])
        self._transforms: Trans[RawMultiSpeakerUtterance, MelMultiSpeakerUtterance] = ChainCall([
            MelMultiSpeakerUtteranceBuilder.create({
                'mel_calcer': make_instance(SpectrogramBuilder, transformer_train_config['features_config']),
            }, dictionary),
        ])
        self._batch_builder = MultiSpeakerSpeechBatchBuilder(
            audio_duration_for_units=True,
            text_pad=dictionary.pad_id())
        self._device = device

    def __call__(self, wave: Wave) -> MultiSpeakerTexts:
        with tempfile.NamedTemporaryFile() as tmp:
            wavfile.write(tmp.name, wave.sample_rate, wave.data)
            separated = self._sepformer.separate_file(path=tmp.name)[0]
        texts = MultiSpeakerTexts([])
        channels = separated.shape[-1]
        for i in range(channels):
            channel = separated[:, i].cpu().numpy()
            channel = _resample(channel, wave.sample_rate / separated.shape[-1], wave.sample_rate)
            mel_utterance = self._transforms(
                RawMultiSpeakerUtterance(
                    MultiSpeakerTexts([]),
                    Wave(wave.sample_rate, channel),
                    ''))
            batch: SpeechBatch = self._batch_builder([mel_utterance])
            batch.to(self._device)
            texts.append(self._generator(batch)[0][0])
        return texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument('--train_config_path', type=str,
                        default='experiments/configs/train_configs/transformer.json')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to Transformer speech recognition model checkpoint')
    parser.add_argument('--beam_size', type=int, default=5)

    # data config
    parser.add_argument('--librispeech_path', type=str, default='LibriSpeech')
    parser.add_argument('--text_processor_config_path', type=str,
                        default='experiments/configs/text_processors/multispeaker_eng.json')

    parser.add_argument('--output_table', type=str, required=True, help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    train_config = parse_train_config(args.train_config_path)
    sepformer_baseline = SepformerInference(args.checkpoint_path, train_config, args.beam_size, args.device)

    datamodule = TransformerDataModule(train_config)
    test_sources = datamodule.configure_test_sources()
    with open(args.text_processor_config_path) as f:
        text_processor: MultiSpeakerTextProcessor = make_instance(Trans, json.load(f))

    results = {}
    for name, test_source in zip(datamodule.validation_table_names, test_sources):
        print('Evaluating on', name)
        recognitions = {}
        wer = SpeakerIndependentWer()
        for sample in tqdm(test_source):
            reference = text_processor.process_text(sample.text)
            hypothesis = sepformer_baseline(sample.audio)
            recognitions[sample.uuid] = {'reference': reference, 'hypothesis': hypothesis}
            wer.update([hypothesis], [reference])
        print('WER:', wer.compute().item())
        results[name] = {
            'cpWER': wer.compute().item(),
            'recognitions': recognitions,
        }

    with open(args.output_table, 'w') as f:
        json.dump(results, f)
