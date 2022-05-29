from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_modules.schedulers.scheduler import LrScheduler


class ReduceLROnPlateuScheduler(LrScheduler, ReduceLROnPlateau):
    pass
