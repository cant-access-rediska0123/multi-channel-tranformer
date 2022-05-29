import abc
import heapq
from typing import Callable, Generic, List, NewType, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data.asr.batch.speech_batch import SpeechBatch
from data.text.dictionary.sentencepiece_dictionary import SentencePieceDictionary
from data.text.text import MultiSpeakerTexts, Text, TokenizedText
from models.generator.generator import BatchType, DiarizationTextGenerator, TextGenerator
from models.text_decoder.text_decoder import TextDecoder
from torch_modules.models.diarization_transformer.multispeaker_decoder import MultiSpeakerDecoder
from torch_modules.models.transformer.encoder_decoder import Decoder, DecoderResult, Encoder, EncoderResult

NormFunction = NewType('NormFunction', Callable[[int], float])


class PowNorm:
    def __init__(self, pow: float = 1.0):
        self._pow = pow

    def __call__(self, x):
        return x ** self._pow

    def __str__(self):
        return f'pow_norm({self._pow})'


class _BeamHypothesis:
    def __init__(self, score: float, tokens: torch.Tensor):
        self.score = score
        self.tokens = tokens

    def __lt__(self, other):
        return self.score < other.score


class _BeamHeap:
    def __init__(self, text_decoder: TextDecoder[SentencePieceDictionary], speakers_num: int, max_hyp_len: int,
                 beam_size: int, beam_norm: NormFunction):
        self._text_decoder = text_decoder
        self._speakers_num = speakers_num
        self._max_hyp_len = max_hyp_len
        self._beam_size = beam_size
        self._beam_norm = beam_norm
        self._partial_hypotheses: List[_BeamHypothesis] = \
            [_BeamHypothesis(0, torch.full((speakers_num, 1), text_decoder.dictionary.bos_id()))]
        self._final_hypothesis: Optional[_BeamHypothesis] = None

    def get_best_hypothesis(self) -> Optional[_BeamHypothesis]:
        if len(self._partial_hypotheses) == 0:
            return None
        return heapq.heappop(self._partial_hypotheses)

    def add_new_hypothesis(self, hyp: _BeamHypothesis):
        if self._final_hypothesis is not None and self._final_hypothesis.score < hyp.score:
            return
        if torch.all(torch.any(hyp.tokens == self._text_decoder.dictionary.eos_id(), dim=1)) or \
                hyp.tokens.size(1) >= self._max_hyp_len:
            if self._final_hypothesis is None or self._final_hypothesis.score > hyp.score:
                self._final_hypothesis = hyp
            return

        heapq.heappush(self._partial_hypotheses, hyp)
        self._partial_hypotheses = heapq.nsmallest(self._beam_size, self._partial_hypotheses)
        heapq.heapify(self._partial_hypotheses)

    def get_final_hypothesis(self) -> Optional[_BeamHypothesis]:
        return self._final_hypothesis

    def __len__(self):
        return len(self._partial_hypotheses)


def _process_next_logits(logits: torch.Tensor, beam_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    speakers_num = logits.shape[0]
    logprobs = F.log_softmax(logits, dim=1).cpu()  # (Sp, *)
    topk_logprobs = logprobs.topk(beam_size, dim=1)  # (Sp, bs)
    combined_topk_logprobs = topk_logprobs.values[0, :]  # (bs,)
    combined_topk_tokens = topk_logprobs.indices[0, :].unsqueeze(1)  # (bs, 1)
    for speaker_id in range(1, speakers_num):
        topk_speaker_logprobs, topk_speaker_tokens = topk_logprobs.values[speaker_id, :], \
                                                     topk_logprobs.indices[speaker_id, :]
        combined_topk_logprobs = torch.FloatTensor(
            np.add.outer(combined_topk_logprobs, topk_speaker_logprobs))  # (bs, bs)

        vals, indices = torch.topk(torch.flatten(combined_topk_logprobs), beam_size)
        topk_indices = np.array(np.unravel_index(indices.numpy(), combined_topk_logprobs.shape))
        prev_indices, new_indices = topk_indices

        prev_tokens = combined_topk_tokens[prev_indices, :]  # (bs, s)
        new_tokens = topk_speaker_tokens[new_indices].unsqueeze(1)  # (bs, 1)
        combined_topk_tokens = torch.cat((prev_tokens, new_tokens), dim=1)  # (bs, s + 1)
        combined_topk_logprobs = vals
    return combined_topk_logprobs, combined_topk_tokens


def _construct_tokens_batch(dictionary: SentencePieceDictionary,
                            hypotheses: List[torch.Tensor]) -> torch.Tensor:
    batch_size = len(hypotheses)
    speakers_num = hypotheses[0].size(0)
    max_length = max(h.size(1) for h in hypotheses)
    prev_tokens = torch.full((batch_size, speakers_num, max_length), dictionary.bos_id())
    for i in range(len(hypotheses)):
        prev_tokens[i, :, :hypotheses[i].size(1)] = hypotheses[i]
    return prev_tokens


class _BeamGeneratorBase(Generic[BatchType]):
    def __init__(self, text_decoder: TextDecoder[SentencePieceDictionary], speakers_num: int,
                 beam_size: int, beam_norm: NormFunction, max_n_steps: Optional[int] = None):
        self._text_decoder = text_decoder
        self._speakers_num = speakers_num
        self._beam_size = beam_size
        self._beam_norm = beam_norm
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
        heaps = [_BeamHeap(self._text_decoder, speakers_num, max_n_steps,
                           self._beam_size, self._beam_norm) for _ in range(batch_size)]

        while any(len(h) > 0 for h in heaps):
            prev_hyps: List[Optional[_BeamHypothesis]] = []
            prev_tokens: List[torch.Tensor] = []
            prev_tokens_lengths: List[int] = []
            for i in range(batch_size):
                hyp = heaps[i].get_best_hypothesis()
                prev_hyps.append(hyp)
                prev_tokens.append(hyp.tokens if hyp is not None else
                                   torch.full((speakers_num, 1), self._text_decoder.dictionary.bos_id()))
                prev_tokens_lengths.append(prev_tokens[-1].size(1))

            prev_tokens_tensor = _construct_tokens_batch(self._text_decoder.dictionary, prev_tokens).to(device)
            decoder_output = self._decoder_call(encoder_result, prev_tokens_tensor).output.cpu().type(torch.float32)

            for i in range(batch_size):
                if prev_hyps[i] is None:
                    continue
                output = decoder_output[i, :, prev_tokens_lengths[i] - 1, :]
                topk_logprobs, topk_tokens = _process_next_logits(output.detach().cpu(), self._beam_size)

                for tokens_score, speaker_tokens in zip(topk_logprobs, topk_tokens):
                    l = prev_tokens_lengths[i]
                    old_denorm_score = prev_hyps[i].score * self._beam_norm(l)
                    new_score = (old_denorm_score - tokens_score) / self._beam_norm(l + 1)

                    new_hypotheses = torch.cat((prev_tokens[i], speaker_tokens.unsqueeze(1)), dim=1)
                    heaps[i].add_new_hypothesis(_BeamHypothesis(new_score, new_hypotheses))

        final_hyps: List[MultiSpeakerTexts] = []
        for h in heaps:
            hyp: Optional[_BeamHypothesis] = h.get_final_hypothesis()
            assert hyp is not None
            final_hyps.append(
                self._text_decoder([TokenizedText(speaker_tokens.tolist()) for speaker_tokens in hyp.tokens]))
        return final_hyps

    def __str__(self) -> str:
        return f'beam_{self._beam_size}_{self._beam_norm}'


class BeamDiarizationGenerator(_BeamGeneratorBase[SpeechBatch], DiarizationTextGenerator):
    def __init__(self, encoder: Encoder, decoder: MultiSpeakerDecoder, speakers_num: int,
                 text_decoder: TextDecoder[SentencePieceDictionary], beam_size: int,
                 beam_norm: NormFunction = PowNorm(), max_n_steps: Optional[int] = None):
        self._encoder = encoder
        self._decoder = decoder
        self._encoder.eval()
        self._decoder.eval()
        super(BeamDiarizationGenerator, self).__init__(text_decoder, speakers_num, beam_size, beam_norm, max_n_steps)

    def _encoder_call(self, features: torch.Tensor, features_lengths: torch.Tensor) -> EncoderResult:
        with torch.no_grad():
            return self._encoder(features, features_lengths)

    def _decoder_call(self, encoder_result: EncoderResult, prev_tokens_tensor: torch.Tensor) -> DecoderResult:
        with torch.no_grad():
            return self._decoder(encoder_result=encoder_result, texts=prev_tokens_tensor, text_lengths=None)


class BeamGenerator(_BeamGeneratorBase[SpeechBatch], TextGenerator):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 text_decoder: TextDecoder[SentencePieceDictionary], beam_size: int,
                 beam_norm: NormFunction = PowNorm(), max_n_steps: Optional[int] = None):
        self._encoder = encoder
        self._decoder = decoder
        self._encoder.eval()
        self._decoder.eval()
        super(BeamGenerator, self).__init__(text_decoder, 1, beam_size, beam_norm, max_n_steps)

    def _encoder_call(self, features: torch.Tensor, features_lengths: torch.Tensor) -> EncoderResult:
        with torch.no_grad():
            return self._encoder(features, features_lengths)

    def _decoder_call(self, encoder_result: EncoderResult, prev_tokens_tensor: torch.Tensor) -> DecoderResult:
        with torch.no_grad():
            decoder_res: DecoderResult = self._decoder(encoder_result=encoder_result,
                                                       texts=prev_tokens_tensor[:, 0, :],
                                                       text_lengths=None)
        return DecoderResult(decoder_res.output.unsqueeze(1))

    def __call__(self, batch: SpeechBatch) -> List[Text]:
        tokens: List[List[Text]] = super(BeamGenerator, self).__call__(batch)  # (B, Sp)
        assert all(len(t) == 1 for t in tokens)
        return [t[0] for t in tokens]
