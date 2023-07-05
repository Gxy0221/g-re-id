from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """  """

from .vru import VRU
from .veriuav import VeRiUAV

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
