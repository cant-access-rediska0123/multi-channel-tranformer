import logging
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, Iterable, List, Tuple, Type

from tqdm import tqdm

from data.asr.datasource.source import Source
from data.asr.streaming_dataset.streaming_dataset import StreamingDataset
from data.text.text import Text
from data.text.text_processor.text_processor import TextProcessor
from data.trans import Trans
from streaming.data.bits import BitInfo, SampleInfo
from streaming.data.bits_batch import BitsBatchBuilder
from streaming.data.bits_batcher import BitsBatcher
from streaming.data.split_transforms import SplitTransforms
from streaming.graph.streaming_graph import MultiSpeakerStreamingGraph, StreamingGraphInterface
from streaming.graph.streaming_graph_parameters import StreamingGraphParameters
from streaming.recognition_model.recognition_model import RecognitionModel, RecognitionModelOutput
from torch_modules.metrics.speaker_independent_wer import SpeakerIndependentWer

LOG = logging.getLogger()


def _run_streaming_graph(t: Tuple[List[BitInfo], List[RecognitionModelOutput]],
                         params: StreamingGraphParameters,
                         streaming_graph_factory: Type[StreamingGraphInterface]) -> List[Text]:
    info, sample_hypotheses = t
    st: StreamingGraphInterface = streaming_graph_factory(params)
    for i, window_hypotheses in enumerate(sample_hypotheses):
        st.add_hypothesis_window(
            window_hypotheses,
            window_start_ms=info[i].start_ms,
            window_end_ms=info[i].end_ms,
            is_first_hypothesis=(i == 0),
            is_last_hypothesis=(i == len(sample_hypotheses) - 1))
    return st.calculate_text_hypothesis()


class StreamingEvaluator:
    def __init__(self,
                 data_source: Source,
                 models_transforms: List[Trans],  # transform batches before model.evaluate
                 text_processor: TextProcessor,
                 model: RecognitionModel,
                 params: StreamingGraphParameters,
                 batch_size: int,
                 max_streaming_graphs_parallel: int,
                 pool_size: int,
                 features_pad: float = 0.0,
                 streaming_graph_factory: Type[StreamingGraphInterface] = MultiSpeakerStreamingGraph):
        self._model = model
        self._params = params
        self._data_source = data_source

        self._datasets = []
        for tr in models_transforms:
            self._datasets.append(StreamingDataset(
                SplitTransforms(tr, text_processor, params.window_size_ms, params.window_shift_ms),
                BitsBatcher(batch_size, 1, BitsBatchBuilder(features_pad=features_pad)),
                data_source))

        self._pool = Pool(pool_size)
        self._max_streaming_graphs_parallel = max_streaming_graphs_parallel
        self._wer_calcer = SpeakerIndependentWer()
        self._streaming_graph_factory = streaming_graph_factory

    def _recognize_bits(self) -> Iterable[Tuple[BitInfo, RecognitionModelOutput]]:
        for batches in zip(*self._datasets):
            LOG.info('Evaluating alignment model...')
            print('Evaluating alignment model...')
            hypotheses: List[RecognitionModelOutput] = self._model(batches)
            for i, hypothesis in enumerate(hypotheses):
                yield batches[0].info[i], hypothesis

    def _merge_bits_recognitions(self, bits_info: Dict[str, List[BitInfo]], processed_samples: List[str],
                                 bits_recognitions: Dict[str, List[RecognitionModelOutput]]) -> List[List[Text]]:
        LOG.info(f'Running streaming graph for samples: {processed_samples}')
        print(f'Running streaming graph for samples: {processed_samples}')
        info = [bits_info[s] for s in processed_samples]
        recognitions: List[List[RecognitionModelOutput]] = [bits_recognitions[s] for s in processed_samples]
        return list(tqdm(self._pool.imap(partial(_run_streaming_graph,
                                                 params=self._params,
                                                 streaming_graph_factory=self._streaming_graph_factory),
                                         zip(info, recognitions)),
                         total=len(processed_samples)))

    def evaluate(self) -> Iterable[Tuple[SampleInfo, List[Text]]]:  # (sample_uuid, hypothesis)
        bits_recognitions: Dict[str, List[RecognitionModelOutput]] = defaultdict(list)
        bits_info: Dict[str, List[BitInfo]] = defaultdict(list)

        processed_samples: List[str] = []
        for bit_info, sample_hypotheses in self._recognize_bits():
            bits_recognitions[bit_info.sample_info.orig_uuid].append(sample_hypotheses)
            bits_info[bit_info.sample_info.orig_uuid].append(bit_info)

            if bit_info.end_ms != bit_info.sample_info.total_ms:
                continue
            processed_samples.append(bit_info.sample_info.orig_uuid)
            if len(processed_samples) < self._max_streaming_graphs_parallel:
                continue

            hypotheses = self._merge_bits_recognitions(bits_info, processed_samples, bits_recognitions)
            for hyp, s in zip(hypotheses, processed_samples):
                yield bits_info[s][0].sample_info, hyp
            for sid in processed_samples:
                del bits_recognitions[sid]
                del bits_info[sid]
            processed_samples = []

        processed_samples = list(bits_info.keys())
        hypotheses = self._merge_bits_recognitions(bits_info, processed_samples, bits_recognitions)
        for hyp, s in zip(hypotheses, processed_samples):
            yield bits_info[s][0].sample_info, hyp

    def calculate_wer(self) -> Dict:
        references, hypotheses = [], []
        result = {}
        for sample_info, sample_hyp in tqdm(
                self.evaluate(),
                total=len(self._data_source) if hasattr(self._data_source, '__len__') else None):
            reference = [Text(r.lower()) for r in sample_info.orig_texts]
            hypothesis = [Text(h.lower()) for h in sample_hyp]
            references.append([Text(r.lower()) for r in sample_info.orig_texts])
            hypotheses.append([Text(h.lower()) for h in sample_hyp])
            wer = self._wer_calcer([reference], [hypothesis]).item()
            result[sample_info.orig_uuid] = {
                'hypothesis': hypothesis,
                'reference': reference,
                'cpWER': wer,
            }
        wer = self._wer_calcer(references, hypotheses).item()
        result['Mean cpWER'] = wer
        return result
