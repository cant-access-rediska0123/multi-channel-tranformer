from typing import List

import torch
from torchmetrics import Metric


def levenshtein(a: List, b: List) -> int:
    """Calculates the Levenshtein distance between a and b.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


# Total WER
class Wer(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.add_state("num", default=torch.tensor(0., dtype=torch.float32, device=device), dist_reduce_fx="sum")
        self.add_state("denum", default=torch.tensor(0., dtype=torch.float32, device=device), dist_reduce_fx="sum")

    def update(self, hypotheses: List[str], references: List[str]):
        if len(hypotheses) != len(references):
            raise ValueError("In word error rate calculation, hypotheses and reference"
                             " lists must have the same number of elements. But I got:"
                             "{0} and {1} correspondingly".format(len(hypotheses), len(references)))
        for h, r in zip(hypotheses, references):
            h_list = h.split()
            r_list = r.split()
            words_right = len(r_list)
            words_left = len(h_list)
            denum = max(words_left, words_right)
            self.num += levenshtein(h_list, r_list)
            self.denum += denum

    def compute(self):
        denum = self.denum.float()
        if denum > 0:
            return self.num.float() / denum
        else:
            return 0
