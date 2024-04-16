# ruff: noqa: F401
from .cutout import Cutout
from .mixup import (
    Mixup,
    MixupIO,
    RegMixup,
    WarpingMixup,
    MixupTO,
    QuantileMixup,
    MITMixup,
    RankMixup_MNDCG,
)
from .transforms import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    MIMOBatchFormat,
    Posterize,
    RepeatTarget,
    Rotation,
    Sharpness,
    Shear,
    Solarize,
    Translate,
)

augmentations = [
    AutoContrast,
    Equalize,
    Posterize,
    Rotation,
    Solarize,
    Shear,
    Translate,
    Contrast,
    Brightness,
    Color,
    Sharpness,
]
