import torch
from reid.utils.registry import Registry
META_ARCH_REGISTRY = Registry("META_ARCH")  
META_ARCH_REGISTRY.__doc__ = """
build_model
"""


def build_model(cfg):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
