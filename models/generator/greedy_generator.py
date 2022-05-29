import abc
from typing import Generic, List, Optional

import torch

from data.asr.batch.speech_batch import SpeechBatch
from data.text.dictionary.sentencepiece_dictionary import SentencePieceDictionary
from data.text.text import MultiSpeakerTexts, Text, TokenizedText
from models.generator.generator import BatchType, DiarizationTextGenerator, TextGenerator
from models.text_decoder.text_decoder import TextDecoder
from torch_modules.models.diarization_transformer.multispeaker_decoder import MultiSpeakerDecoder
from torch_modules.models.transformer.encoder_decoder import Decoder, DecoderResult, Encoder, EncoderResult


class _GreedyGeneratorBase(Generic[BatchType]):
    def __init__(self, text_decoder: TextDecoder[SentencePieceDictionary], speakers_num: int,
                 max_n_steps: Optional[int]):
        self._text_decoder = text_decoder
        self._speakers_num = speakers_num
        self._max_n_steps = max_n_steps

    @abc.abstractmethod
    def _encoder_call(self, features: torch.Tensor, features_lengths: torch.Tensor) -> EncoderResult:
        pass

    @abc.abstractmethod
    def _decoder_call(self, encoder_result: EncoderResult, prev_tokens_tensor: torch.Tensor) -> DecoderResult:
        pass

    def __call__(self, batch: BatchType) -> List[MultiSpeakerTexts]:  # (B, Sp)
        batch_size, speakers_num, max_n_steps = batch.features.size(0), self._speakers_num, batch.features.size(2) - 1
        if self._max_n_steps is not None:
            max_n_steps = self._max_n_steps
        device = batch.device

        encoder_result: EncoderResult = self._encoder_call(batch.features, batch.features_lengths)

        prev_tokens_tensor = torch.full((batch_size, speakers_num, 1),
                                        self._text_decoder.dictionary.bos_id()).long().to(device)
        lengths = torch.full((batch_size, speakers_num), -1).to(device)

        for step in range(max_n_steps):
            decoder_result: DecoderResult = self._decoder_call(encoder_result, prev_tokens_tensor)
            next_tokens = decoder_result.output[:, :, -1, :].argmax(dim=2)
            prev_tokens_tensor = torch.cat((prev_tokens_tensor, next_tokens.unsqueeze(2)), 2)
            lengths[torch.logical_and(lengths == -1,
                                      next_tokens == self._text_decoder.dictionary.eos_id())] = prev_tokens_tensor.size(
                2)
            if torch.count_nonzero(lengths == -1) == 0:
                break
        lengths[lengths == -1] = max_n_steps + 1

        res = [self._text_decoder(
            [TokenizedText(prev_tokens_tensor[i, sp, :lengths[i, sp]].cpu().tolist())
             for sp in range(speakers_num)])
            for i in range(batch_size)]

        return res

    def __str__(self) -> str:
        return 'greedy'


class GreedyDiarizationGenerator(_GreedyGeneratorBase[SpeechBatch], DiarizationTextGenerator):
    def __init__(self, encoder: Encoder, decoder: MultiSpeakerDecoder, speakers_num: int,
                 text_decoder: TextDecoder[SentencePieceDictionary], max_n_steps: Optional[int] = None):
        self._encoder = encoder
        self._decoder = decoder
        self._encoder.eval()
        self._decoder.eval()
        super(GreedyDiarizationGenerator, self).__init__(text_decoder, speakers_num, max_n_steps)

    def _encoder_call(self, features: torch.Tensor, features_lengths: torch.Tensor) -> EncoderResult:
        with torch.no_grad():
            return self._encoder(features, features_lengths)

    def _decoder_call(self, encoder_result: EncoderResult, prev_tokens_tensor: torch.Tensor) -> DecoderResult:
        with torch.no_grad():
            return self._decoder(encoder_result=encoder_result, texts=prev_tokens_tensor, text_lengths=None)


class GreedyGenerator(_GreedyGeneratorBase[SpeechBatch], TextGenerator):
    def __init__(self, encoder: Encoder, decoder: Decoder, text_decoder: TextDecoder[SentencePieceDictionary],
                 max_n_steps: Optional[int] = None):
        self._encoder = encoder
        self._decoder = decoder
        self._encoder.eval()
        self._decoder.eval()
        super(GreedyGenerator, self).__init__(text_decoder, 1, max_n_steps)

    def _encoder_call(self, features: torch.Tensor, features_lengths: torch.Tensor) -> EncoderResult:
        with torch.no_grad():
            return self._encoder(features.cuda(), features_lengths.cuda())

    def _decoder_call(self, encoder_result: EncoderResult, prev_tokens_tensor: torch.Tensor) -> DecoderResult:
        with torch.no_grad():
            decoder_res: DecoderResult = self._decoder(encoder_result=encoder_result,
                                                       texts=prev_tokens_tensor[:, 0, :].cuda(),
                                                       text_lengths=None)
        return DecoderResult(decoder_res.output.unsqueeze(1))

    def __call__(self, batch: SpeechBatch) -> List[Text]:
        tokens: List[MultiSpeakerTexts] = super(GreedyGenerator, self).__call__(batch)  # (B, Sp)
        assert all(len(t) == 1 for t in tokens)
        return [t[0] for t in tokens]
