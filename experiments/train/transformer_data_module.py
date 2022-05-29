import math
from typing import Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.asr.batch.speech_batch import SpeechBatch
from data.asr.batch_builder.multispeaker_speech_batch_builder import MultiSpeakerSpeechBatchBuilder
from data.asr.datasource.libri_speech.source import LibriSpeechInfo, TrainLibriSpeechSource
from data.asr.datasource.multispeaker_datasource import MultiSpeakerSource
from data.asr.datasource.test_libri_speech.source import SyntheticGenParams, TestMultiSpeakerLibriSpeechSource, \
    TestSingleSpeakerLibriSpeechSource
from data.asr.datasource.weighted_multi_datasource import Source, WeightedMultiDataSource
from data.asr.streaming_dataset.block_batcher import BlockBatcher
from data.asr.streaming_dataset.streaming_dataset import StreamingDataset
from data.asr.transforms.mel_builder.multispeaker_mel_builder import MelMultiSpeakerUtteranceBuilder
from data.asr.transforms.multispeaker.flatten_text_transform import FlattenTexts
from data.asr.transforms.multispeaker.utterance_mixer import UtteranceMixer
from data.asr.transforms.spectrogram_builder.spectrogram_builder import SpectrogramBuilder
from data.text.dictionary.sentencepiece_dictionary import Dictionary, SentencePieceDictionary
from data.text.text_processor.text_processor import TextProcessor
from data.trans import ChainCall, Trans
from data.utterance.utterance import MelMultiSpeakerUtterance, RawMultiSpeakerUtterance, RawUtterance
from factory.factory import make_instance


def _collate_fn(x):
    return x[0]


def _sort_fn(sample):
    return sample.audio.duration_secs


class TransformerDataModule(pl.LightningDataModule):
    def __init__(self, train_config: Dict):
        super().__init__()
        self._train_dataset = None
        self._val_datasets = None
        self._test_datasets = None

        self._resources_config = train_config['resources_config']
        self._train_config = train_config['data_config']['train']
        self._val_configs = train_config['data_config']['validation']
        self._test_configs = train_config['data_config'].get('test', None)
        self._text_processor_config = train_config['text_processor_config']
        self._dictionary_config = train_config['dictionary_config']
        self._features_config = train_config['features_config']
        self._speakers = len(self._train_config['speaker_num_freqs'])

        self._val_table_names = [val_config['name'] for val_config in self._val_configs]
        self._test_table_names = []
        if self._test_configs is not None:
            self._test_table_names = [test_config['name'] for test_config in self._test_configs]

    def construct_transforms(self, wave_augmentations: Optional[List] = None,
                             spec_augmentations: Optional[List] = None,
                             mix_audios: bool = False) -> Trans[List[RawUtterance], MelMultiSpeakerUtterance]:
        transforms = []
        if mix_audios:
            transforms.append(UtteranceMixer(speakers_num=self._speakers))
        text_processor: TextProcessor = make_instance(Trans, self._text_processor_config)
        transforms.append(text_processor)
        dictionary: SentencePieceDictionary = make_instance(Dictionary, self._dictionary_config)
        transforms.append(MelMultiSpeakerUtteranceBuilder.create({
            'wave_augmentations': wave_augmentations if wave_augmentations else [],
            'spec_augmentations': spec_augmentations if spec_augmentations else [],
            'mel_calcer': make_instance(SpectrogramBuilder, self._features_config),
        }, dictionary))
        if self._train_config.get('flatten_texts', False):
            transforms.append(FlattenTexts(dictionary))
        return ChainCall(transforms)

    def _configure_block_batcher(self, batch_size, block_size) -> BlockBatcher:
        dictionary: SentencePieceDictionary = make_instance(Dictionary, self._dictionary_config)
        return BlockBatcher(
            batch_size=batch_size,
            block_size=block_size,
            batch_builder=MultiSpeakerSpeechBatchBuilder(
                audio_duration_for_units=True,
                text_pad=dictionary.pad_id(),
                features_pad=0.0,
            ),
            sort_by=_sort_fn)

    def configure_train_source(self) -> Source[List[RawUtterance]]:
        sources: List[Source[RawUtterance]] = []
        weights: List[float] = []
        for table in self._train_config['tables']:
            table_info = LibriSpeechInfo(table)
            sources.append(TrainLibriSpeechSource(table_info))
            weights.append(table_info.weight)

        speaker_num_freqs = self._train_config['speaker_num_freqs']
        multispeaker_sources: List[Source[List[RawUtterance]]] = []
        for speakers_num in range(1, len(speaker_num_freqs) + 1):
            multispeaker_sources.append(
                MultiSpeakerSource(
                    WeightedMultiDataSource[RawUtterance](sources, weights),
                    speakers_num=speakers_num))

        train_source = WeightedMultiDataSource[List[RawUtterance]](multispeaker_sources, speaker_num_freqs)
        return train_source

    def configure_train_dataset(self) -> StreamingDataset[SpeechBatch]:
        return StreamingDataset[SpeechBatch](
            self.construct_transforms(
                wave_augmentations=self._train_config['wave_augmentations'],
                spec_augmentations=self._train_config['spec_augmentations'],
                mix_audios=True),
            self._configure_block_batcher(
                batch_size=self._resources_config['batch_size'],
                block_size=self._resources_config['block_size']),
            self.configure_train_source())

    def _configure_val_sources(self, configs: Optional[List[Dict]]) -> List[Source[RawMultiSpeakerUtterance]]:
        if configs is None:
            return []
        sources = []
        for config in configs:
            if config['speakers_num'] == 1:
                source = TestSingleSpeakerLibriSpeechSource(source_info=LibriSpeechInfo(config['librispeech_path']),
                                                            total_speakers=self._speakers)
            else:
                source = TestMultiSpeakerLibriSpeechSource(
                    librispeech_dir=config['librispeech_path'],
                    sample_pairs_config=config['sample_pairs_config'],
                    params=SyntheticGenParams(speakers_num=config['speakers_num'], overlap=config['overlap']))
            sources.append(source)
        return sources

    def configure_test_sources(self) -> List[Source[RawMultiSpeakerUtterance]]:
        return self._configure_val_sources(self._test_configs)

    def _configure_val_datasets(self, configs: Optional[List[Dict]]) -> List[StreamingDataset[SpeechBatch]]:
        sources = self._configure_val_sources(configs)
        batch_size = self._resources_config['batch_size']
        return [StreamingDataset[SpeechBatch](
            self.construct_transforms(),
            self._configure_block_batcher(batch_size, 1),
            val_source,
            size=math.ceil(len(val_source) / batch_size)) for val_source in sources]

    def train_dataloader(self) -> DataLoader:
        self._train_dataset = self.configure_train_dataset()
        return DataLoader(
            self._train_dataset,
            num_workers=self._resources_config['num_workers'],
            pin_memory=True,
            prefetch_factor=self._resources_config['block_size'] * 2,
            collate_fn=_collate_fn)

    def val_dataloader(self) -> List[DataLoader]:
        self._val_datasets = self._configure_val_datasets(self._val_configs)
        return [DataLoader(val_dataset, num_workers=0, pin_memory=True, collate_fn=_collate_fn)
                for val_dataset in self._val_datasets]

    def test_dataloader(self) -> List[DataLoader]:
        self._test_datasets = self._configure_val_datasets(self._test_configs)
        return [DataLoader(val_dataset, num_workers=0, pin_memory=True, collate_fn=_collate_fn)
                for val_dataset in self._test_datasets]

    @property
    def validation_table_names(self) -> List[str]:
        return self._val_table_names

    @property
    def test_table_names(self) -> List[str]:
        return self._test_table_names
