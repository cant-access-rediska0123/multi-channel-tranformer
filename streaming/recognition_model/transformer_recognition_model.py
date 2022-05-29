import logging
from typing import List, Tuple

import numpy as np

from aligner.time_ctc_aligner import TimeCtcAligner
from data.text.dictionary.ctc_dictionary import CtcDictionary
from data.text.text import Text
from models.generator.generator import DiarizationTextGenerator
from models.nemo_jasper import NemoJasper
from streaming.recognition_model.recognition_model import RecognitionModel, RecognitionModelOutput

LOG = logging.getLogger()


class TransformerRecognitionModel(RecognitionModel):
    def __init__(self,
                 text_generator: DiarizationTextGenerator,
                 jasper: NemoJasper,
                 ctc_dictionary: CtcDictionary,
                 ms_in_frame: int,
                 pool_size: int,
                 use_joint_speakers_alignment: bool = False):
        self._text_generator = text_generator
        self._jasper = jasper
        self._ms_in_frame = ms_in_frame
        self._time_aligner = TimeCtcAligner(ctc_dictionary, pool_size, ms_in_frame, use_joint_speakers_alignment)

    def _run_recognition(self, text_generator_batch, ctc_model_batch) -> Tuple[List[List[Text]], np.ndarray]:
        text_generator_batch.cuda()
        hypotheses = self._text_generator(text_generator_batch)
        text_generator_batch.cpu()
        ctc_model_batch.cuda()
        ctc_logits = self._jasper(ctc_model_batch.features, ctc_model_batch.features_lengths).cpu().numpy()
        return hypotheses, ctc_logits

    def __call__(self, batches: Tuple) -> List[RecognitionModelOutput]:
        LOG.info('Recognizing texts...')
        print('Recognizing texts...')
        if len(batches) != 2:
            raise Exception(f'Expected transformer and ctc batch, got {len(batches)} batches')
        text_generator_batch, ctc_model_batch = batches
        hypotheses, ctc_logits = self._run_recognition(text_generator_batch, ctc_model_batch)
        LOG.info('Aligning hypotheses...')
        print('Aligning hypotheses...')
        alignments = self._time_aligner(hypotheses, ctc_logits)

        return [RecognitionModelOutput(a, l) for a, l in zip(alignments, ctc_logits)]
