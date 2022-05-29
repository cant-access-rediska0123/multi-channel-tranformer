from data.asr.transforms.augmentations.spectrogram_augmentations import SpecAugment, SpectrogramAugmetation
from data.asr.transforms.augmentations.wave_augmentations import GainAugmentation, SpeedAugmentation, WaveAugmentation
from factory.factory import Factory

Factory.register(WaveAugmentation, {
    "speed_wave_augmentation": SpeedAugmentation,
    "gain_wave_augmentation": GainAugmentation,
})

Factory.register(SpectrogramAugmetation, {
    "spec_augmentation": SpecAugment,
})
