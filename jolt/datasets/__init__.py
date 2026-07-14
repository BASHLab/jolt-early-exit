"""Unified dataset loaders. Paths are passed in via config, never hardcoded."""

from .cifar import cifar_dataloaders, cifar_num_classes
from .imagenette import imagenette_dataloaders
from .pamap2 import pamap2_dataloaders
from .tiny_imagenet import tiny_imagenet_dataloaders
from .uci_har import uci_har_dataloaders
from .gsc_v2 import gsc_v2_dataloaders

__all__ = ["cifar_dataloaders", "cifar_num_classes", "imagenette_dataloaders",
           "pamap2_dataloaders", "tiny_imagenet_dataloaders", "uci_har_dataloaders",
           "gsc_v2_dataloaders"]
