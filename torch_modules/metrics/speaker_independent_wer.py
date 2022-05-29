from itertools import permutations
from typing import List, Tuple

import torch
from torchmetrics import Metric

from data.text.text import Text
from torch_modules.metrics.wer import levenshtein


def speaker_independent_word_error_rate(hypotheses: List[str], references: List[str]) -> Tuple[float, float, float]:
    if len(hypotheses) != len(references):
        raise ValueError("In word error rate calculation, hypotheses and reference"
                         " lists must have the same number of elements. But I got:"
                         "{0} and {1} correspondingly".format(len(hypotheses), len(references)))

    denum, score = 0, 0
    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        denum += max(len(r_list), len(h_list))
        score += levenshtein(h_list, r_list)

    if denum > 0:
        wer = 1.0 * score / denum
    else:
        wer = 0
    return wer, score, denum


# Total cpWER
class SpeakerIndependentWer(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("num", default=torch.tensor(0., dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("denum", default=torch.tensor(0., dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, hypotheses: List[List[str]], references: List[List[str]]):
        for sample_references, sample_hypotheses in zip(references, hypotheses):
            best_wer, best_sc, best_denum = None, None, None

            # pad with empty speakers
            speakers_num = max(len(sample_hypotheses), len(sample_references))
            sample_hypotheses += [Text('') for _ in range(len(sample_hypotheses), speakers_num)]
            sample_references += [Text('') for _ in range(len(sample_references), speakers_num)]

            for sample_hypotheses_perm in permutations(sample_hypotheses):
                wer, sc, denum = speaker_independent_word_error_rate(list(sample_hypotheses_perm), sample_references)
                if best_wer is None or wer < best_wer:
                    best_wer, best_sc, best_denum = wer, sc, denum
            self.num += best_sc
            self.denum += best_denum

    def compute(self):
        denum = self.denum.float()
        if denum > 0:
            return self.num.float() / denum
        else:
            return self.num.float()
