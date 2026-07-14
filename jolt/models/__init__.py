"""JOLT early-exit model backbones with a shared interface (see :mod:`jolt.models.base`)."""

from .base import ExitModel
from .mobilenetv2_exit import MobileNetV2, mobilenetv2_exit
from .resnet_exit import ResNet, resnet18_exit, resnet34_exit, resnet50_exit
from .resnet56_exit import ResNet56Exit, resnet56_exit, resnet110_exit
from .wideresnet28_2_exit import WideResNet28_2Exit, wideresnet28_2_exit
from .resnet56_sdn_exit import ResNet56SDNExit, resnet56_sdn_exit
from .resnet56_sdn_litelaplace_exit import (
    ResNet56SDNLiteLaplaceExit,
    resnet56_sdn_litelaplace_exit,
)
from .wideresnet28_10_sdn_exit import WideResNet2810SDNExit, wideresnet28_10_sdn_exit
from .wideresnet28_10_sdn_jei_exit import (
    WideResNet2810SDNJEIExit,
    wideresnet28_10_sdn_jei_exit,
)
from .deepconvlstm_exit import DeepConvLSTMExit, deepconvlstm_exit
from .uci_har_1d_cnn_exit import UCIHAR1DCNNExit, uci_har_1d_cnn_exit
from .inceptiontime_exit import InceptionTimeExit, inceptiontime_exit
from .tcn_har_exit import TCNHarExit, tcn_har_exit
from .bcresnet_exit import BCResNet8Exit, bcresnet8_exit
from .cct7_exit import CCT7Exit, cct7_exit
from .mobilenetv3_exit import MobileNetV3LargeExit, mobilenetv3_large_exit
from .mobilenetv3_small_exit import MobileNetV3SmallExit, mobilenetv3_small_exit
from .densenet_bc_100_exit import DenseNetBC100Exit, densenet_bc_100_exit
from .reslstm_exit import ResLSTMExit, reslstm_exit
from .tst_exit import TSTExit, tst_exit
from .convmixer1d_exit import ConvMixer1DExit, convmixer1d_exit
from .convmixer_256_8_exit import ConvMixer256_8Exit, convmixer_256_8_exit
from .mobilevit_xxs_exit import MobileViTXXSExit, mobilevit_xxs_exit
from .bert_early_exit import BertEarlyExit
from .selective import SelectiveExitModel

__all__ = [
    "ExitModel",
    "MobileNetV2",
    "mobilenetv2_exit",
    "ResNet",
    "resnet18_exit",
    "resnet34_exit",
    "resnet50_exit",
    "ResNet56Exit",
    "resnet56_exit",
    "ResNet56SDNExit",
    "resnet56_sdn_exit",
    "ResNet56SDNLiteLaplaceExit",
    "resnet56_sdn_litelaplace_exit",
    "WideResNet2810SDNExit",
    "wideresnet28_10_sdn_exit",
    "WideResNet2810SDNJEIExit",
    "wideresnet28_10_sdn_jei_exit",
    "DeepConvLSTMExit",
    "deepconvlstm_exit",
    "UCIHAR1DCNNExit",
    "uci_har_1d_cnn_exit",
    "InceptionTimeExit",
    "inceptiontime_exit",
    "TCNHarExit",
    "tcn_har_exit",
    "BCResNet8Exit",
    "bcresnet8_exit",
    "CCT7Exit",
    "cct7_exit",
    "BertEarlyExit",
    "SelectiveExitModel",
]
