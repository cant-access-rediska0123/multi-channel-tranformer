import torch
import torch.nn as nn


class PatchedBatchNorm1d(nn.BatchNorm1d):
    def fix_inf(self):
        if self.track_running_stats and not torch.all(torch.isfinite(self.running_mean)):
            self.reset_running_stats()
