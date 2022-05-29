from factory.factory import Factory
from torch_modules.schedulers.reduce_lr_on_plateu import ReduceLROnPlateuScheduler
from torch_modules.schedulers.scheduler import LrScheduler

Factory.register(LrScheduler, {
    "reduce_lr_on_plateu": ReduceLROnPlateuScheduler,
})
