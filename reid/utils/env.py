import torch
import importlib
import importlib.util
import logging
import numpy as np
import os
import random
import sys
from datetime import datetime

__all__ = ["seed_all_rng"]


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

def seed_all_rng(seed=None):
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _configure_libraries():
    disable_cv2 = int(os.environ.get("DETECTRON2_DISABLE_CV2", False))
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ImportError:
            pass

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))

    assert get_version(torch) >= (1, 4), "Requires torch>=1.4"
    import yaml
    assert get_version(yaml) >= (5, 1), "Requires pyyaml>=5.1"


_ENV_SETUP_DONE = False


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()

    custom_module_path = os.environ.get("FASTREID_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        pass


def setup_custom_environment(custom_module):
    if custom_module.endswith(".py"):
        module = _import_file("reid.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)
    module.setup_environment()