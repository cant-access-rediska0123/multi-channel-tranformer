from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.asr.batch.speech_batch import SpeechBatch
from data.text.dictionary import SentencePieceDictionary
from data.text.text import MultiSpeakerTexts, MultiSpeakerTokenizedTexts, TokenizedText
from factory.factory import make_instance
from models.generator.beam_generator import BeamDiarizationGenerator
from models.generator.greedy_generator import DiarizationTextGenerator, GreedyDiarizationGenerator
from models.text_decoder.text_decoder import TextDecoder
from torch_modules.metrics.speaker_independent_wer import SpeakerIndependentWer
from torch_modules.models.diarization_transformer.multispeaker_decoder import MultiSpeakerDecoder
from torch_modules.models.transformer.encoder_decoder import DecoderResult, Encoder, EncoderResult
from torch_modules.objectives.cross_entropy import CrossEntropyLoss
from torch_modules.optimizers.optimizer import Optimizer
from torch_modules.schedulers.scheduler import LrScheduler


class TransformerResult:
    def __init__(self, output: torch.Tensor, encoded_lengths: torch.Tensor):
        self._output = output
        self._encoded_lengths = encoded_lengths
        self._probs = self._log_probs = None

    @property
    def output(self) -> torch.Tensor:
        return self._output

    @property
    def encoded_lengths(self) -> torch.Tensor:
        return self._encoded_lengths

    @property
    def probs(self) -> torch.Tensor:
        if self._probs is None:
            self._probs = F.softmax(self._output, dim=-1)
        return self._probs

    @property
    def log_probs(self) -> torch.Tensor:
        if self._log_probs is None:
            self._log_probs = F.log_softmax(self._output, dim=-1)
        return self._log_probs

    @property
    def predictions(self) -> torch.Tensor:
        return self.log_probs.argmax(dim=-1, keepdim=False).int()


def _get_dataset_name(dataset_names: List[str], dataloader_idx):
    return dataset_names[dataloader_idx] if len(dataset_names) > dataloader_idx else 'None'


@dataclass
class Recognition:
    sample_uuid: str
    hypothesis: MultiSpeakerTexts
    reference: MultiSpeakerTexts

    def to_json(self) -> Dict:
        return {'id': self.sample_uuid, 'hypothesis': self.hypothesis, 'reference': self.reference}


class DiarizationTransformer(pl.LightningModule):
    def __init__(self, dictionary: SentencePieceDictionary, transformer_config: Dict, input_dim: int,
                 optimizer_config: Dict,
                 val_dataset_names: Optional[List[str]] = None,
                 test_dataset_names: Optional[List[str]] = None):
        nn.Module.__init__(self)
        super().__init__()
        self.speakers_num = transformer_config['speakers_num']
        self.encoder = Encoder(transformer_config['encoder'], input_dim, dictionary)
        self.decoder = MultiSpeakerDecoder(
            transformer_config['decoder'], self.encoder.output_dim(), dictionary, speakers_num=self.speakers_num)

        self._objective = CrossEntropyLoss(pad_id=dictionary.pad_id())
        self.dictionary = dictionary
        self.text_decoder = make_instance(TextDecoder, {
            **transformer_config['text_decoder'],
            'dictionary': dictionary})

        self._validation_generators: List[DiarizationTextGenerator] = []
        self._test_generators: List[DiarizationTextGenerator] = [
            GreedyDiarizationGenerator(self.encoder, self.decoder, self.speakers_num,
                                       self.text_decoder),
            BeamDiarizationGenerator(self.encoder, self.decoder, self.speakers_num,
                                     self.text_decoder, beam_size=5)]

        self._optimizer_config = optimizer_config
        self._val_dataset_names = val_dataset_names if val_dataset_names is not None else []
        self._test_dataset_names = test_dataset_names if test_dataset_names is not None else []

        self._train_wer_calcer = SpeakerIndependentWer()
        self._val_wer_calcers = nn.ModuleDict({
            f'{val_dataset_name}_{generator_name}_wer': SpeakerIndependentWer()
            for generator_name in ['teacher-forced'] + [str(g) for g in self._validation_generators]
            for val_dataset_name in self._val_dataset_names})
        self._test_wer_calcers = nn.ModuleDict({
            f'{val_dataset_name}_{generator_name}_wer': SpeakerIndependentWer()
            for generator_name in ['teacher-forced'] + [str(g) for g in self._test_generators]
            for val_dataset_name in self._test_dataset_names})

        self.save_hyperparameters()

    def forward(self, batch: SpeechBatch):
        encoder_result: EncoderResult = self.encoder(batch.features, batch.features_lengths)
        decoder_result: DecoderResult = self.decoder(encoder_result, batch.tokens[:, :, :-1],
                                                     batch.tokens_lengths - 1)  # delete EOS
        return TransformerResult(output=decoder_result.output,
                                 encoded_lengths=encoder_result.encoded_lengths)

    def decode_tokenized_texts(self, tokenized_texts: List[MultiSpeakerTokenizedTexts]) -> List[MultiSpeakerTexts]:
        return [self.text_decoder(texts) for texts in tokenized_texts]

    def _teacher_forced_batch_wer(self, batch: SpeechBatch, result: TransformerResult) -> Tuple:
        hypotheses = [self.text_decoder([
            TokenizedText(t) for t in preds.cpu().tolist()]) for preds in result.predictions]
        references = self.decode_tokenized_texts(batch.texts)
        return hypotheses, references

    def _generator_batch_wer(self, batch: SpeechBatch, generator: DiarizationTextGenerator) -> Tuple:
        hypotheses: List[MultiSpeakerTexts] = generator(batch)
        references: List[MultiSpeakerTexts] = self.decode_tokenized_texts(batch.texts)
        return hypotheses, references

    def _process_recognitions(self, batch: SpeechBatch,
                              hypotheses: List[MultiSpeakerTexts],
                              references: List[MultiSpeakerTexts],
                              mode: str, name: str, generator_name: str, print: bool,
                              collect_recognitions: bool):
        wer_name = f'{name}_{generator_name}_wer'
        if mode == 'train':
            wer = self._train_wer_calcer(references, hypotheses).float()
            self.log(wer_name, wer, on_step=True, on_epoch=False)
        elif mode == 'val':
            self._val_wer_calcers[wer_name](references, hypotheses)
            self.log(wer_name, self._val_wer_calcers[wer_name], on_step=False, on_epoch=True, sync_dist=True)
        elif mode == 'test':
            self._test_wer_calcers[wer_name](references, hypotheses)
            self.log(wer_name, self._test_wer_calcers[wer_name], on_step=False, on_epoch=True, sync_dist=True)
        else:
            raise Exception('Unknown mode:', mode)

        if collect_recognitions:
            assert len(batch.orig_uuids) == len(hypotheses) == len(references)
            for uuid, hyp, ref in zip(batch.orig_uuids, hypotheses, references):
                self.trainer.test_recognitions[f'{name}_{generator_name}'].append(Recognition(uuid, hyp, ref))

        if print:
            self.print('Generator:', generator_name)
            for sp in range(max(len(references[0]), len(hypotheses[0]))):
                self.print(f" Ref{sp + 1}: {references[0][sp] if sp < len(references[0]) else ''}")
                self.print(f" Hyp{sp + 1}: {hypotheses[0][sp] if sp < len(hypotheses[0]) else ''}")
                self.print()

    def _batch_summary(self, batch: SpeechBatch, result: TransformerResult, name: str,
                       generators: List[DiarizationTextGenerator],
                       collect_recognitions: bool,
                       mode: str, print: bool = False):
        if print:
            self.print(f"\n-------------{name} batch summary-------------")
        with torch.no_grad():
            hypotheses, references = self._teacher_forced_batch_wer(batch, result)
        self._process_recognitions(batch, hypotheses, references, name=name,
                                   mode=mode, print=print, generator_name='teacher-forced',
                                   collect_recognitions=collect_recognitions)
        for generator in generators:
            hypotheses, references = self._generator_batch_wer(
                batch, generator)
            self._process_recognitions(batch, hypotheses, references, name=name,
                                       mode=mode, print=print, generator_name=str(generator),
                                       collect_recognitions=collect_recognitions)
        if print:
            self.print(f"-------------============-------------")

    def _step(self, batch: SpeechBatch, generators: List[DiarizationTextGenerator],
              name: str, gen_batch_summary: bool, print: bool, mode: str, collect_recognitions: bool = False):
        result: TransformerResult = self(batch)
        loss = self._objective(logits=result.output, targets=batch.tokens[:, :, 1:])  # delete BOS
        if gen_batch_summary:
            self._batch_summary(batch, result, name=name, generators=generators, mode=mode, print=print,
                                collect_recognitions=collect_recognitions)
        return loss

    def training_step(self, batch: SpeechBatch, batch_idx):
        return self._step(batch, self._validation_generators, name='train',
                          gen_batch_summary=batch_idx % 100 == 0, print=True, mode='train')

    def validation_step(self, batch: SpeechBatch, batch_idx, dataloader_idx=0):
        name = _get_dataset_name(self._val_dataset_names, dataloader_idx)
        return self._step(batch, self._validation_generators, name=name,
                          gen_batch_summary=True, print=batch_idx == 0, mode='val')

    def on_test_epoch_start(self):
        self.trainer.test_recognitions = defaultdict(list)

    def test_step(self, batch: SpeechBatch, batch_idx: int):
        if str(self.device) != 'cpu':
            batch = batch.cuda()
        name = _get_dataset_name(self._test_dataset_names, self.trainer.test_idx)
        return self._step(batch, self._test_generators, name=name,
                          gen_batch_summary=True, print=batch_idx % 100 == 0,
                          mode='test', collect_recognitions=True)

    def configure_optimizers(self):
        optimizer = make_instance(Optimizer, {
            'params': self.parameters(),
            **self._optimizer_config['optimizer'],
        })
        scheduler = make_instance(LrScheduler, {
            'optimizer': optimizer,
            **self._optimizer_config['lr_scheduler'].pop('scheduler')
        })
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                **self._optimizer_config['lr_scheduler']
            }
        }
