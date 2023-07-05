import copy
import logging
import os
from collections import defaultdict
from typing import Any
from typing import Optional, List, Dict, NamedTuple, Tuple, Iterable
import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from torch.nn.parallel import DistributedDataParallel, DataParallel
from reid.utils.file_io import PathManager

class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple]),
        ],
    )
):
    pass


class Checkpointer(object):
    def __init__(
            self,
            model: nn.Module,
            save_dir: str = "",
            *,
            save_to_disk: bool = True,
            **checkpointables: object,
    ):
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables = copy.copy(checkpointables)
        self.logger = logging.getLogger(__name__)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

        self.path_manager = PathManager

    def save(self, name: str, **kwargs: Dict[str, str]):
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with PathManager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> object:
        if not path:
            self.logger.info("No checkpoint found. Training model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (
                incompatible is not None
        ): 
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint: 
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        return checkpoint

    def has_checkpoint(self):      
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return PathManager.exists(save_file)

    def get_checkpoint_file(self):       
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with PathManager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:         
            return ""
        return os.path.join(self.save_dir, last_saved)

    def get_all_checkpoint_files(self):
        all_model_checkpoints = [
            os.path.join(self.save_dir, file)
            for file in PathManager.ls(self.save_dir)
            if PathManager.isfile(os.path.join(self.save_dir, file))
               and file.endswith(".pth")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool = True):
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])

    def tag_last_checkpoint(self, last_filename_basename: str):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with PathManager.open(save_file, "w") as f:
            f.write(last_filename_basename)

    def _load_file(self, f: str):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint: Any):
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)

        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

    def _log_incompatible_keys(self, incompatible: _IncompatibleKeys) -> None:
        for k, shape_checkpoint, shape_model in incompatible.incorrect_shapes:
            self.logger.warning(
                "Skip loading parameter '{}' to the model due to incompatible "
                "shapes: {} in the checkpoint but {} in the "
                "model! You might want to double check if this is expected.".format(
                    k, shape_checkpoint, shape_model
                )
            )
        if incompatible.missing_keys:
            missing_keys = _filter_reused_missing_keys(
                self.model, incompatible.missing_keys
            )
            if missing_keys:
                self.logger.info(get_missing_parameters_message(missing_keys))
        if incompatible.unexpected_keys:
            self.logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    def _convert_ndarray_to_tensor(self, state_dict: dict):
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(
                    v, torch.Tensor
            ):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(
                        k, type(v)
                    )
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)


class PeriodicCheckpointer:
    def __init__(self, checkpointer: Any, period: int, max_epoch: int = None):
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_epoch = max_epoch
        self.best_metric = -1

    def step(self, epoch: int, **kwargs: Any):
        epoch = int(epoch)
        additional_state = {"epoch": epoch}
        additional_state.update(kwargs)
        if (epoch + 1) % self.period == 0 and epoch < self.max_epoch - 1:
            if additional_state["metric"] > self.best_metric:
                self.checkpointer.save(
                    "model_best", **additional_state
                )
                self.best_metric = additional_state["metric"]
            self.checkpointer.save(
                "model_{:04d}".format(epoch), **additional_state
            )
        if epoch >= self.max_epoch - 1:
            if additional_state["metric"] > self.best_metric:
                self.checkpointer.save(
                    "model_best", **additional_state
                )
            self.checkpointer.save("model_final", **additional_state)

    def save(self, name: str, **kwargs: Any):
        self.checkpointer.save(name, **kwargs)


def _filter_reused_missing_keys(model: nn.Module, keys: List[str]) -> List[str]:
    keyset = set(keys)
    param_to_names = defaultdict(set) 
    for module_prefix, module in _named_modules_with_dup(model):
        for name, param in list(module.named_parameters(recurse=False)) + list(
                module.named_buffers(recurse=False) 
        ):
            full_name = (module_prefix + "." if module_prefix else "") + name
            param_to_names[param].add(full_name)
    for names in param_to_names.values():
        if any(n in keyset for n in names) and not all(n in keyset for n in names):
            [keyset.remove(n) for n in names if n in keyset]
    return list(keyset)


def get_missing_parameters_message(keys: List[str]) -> str:
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)
    try:
        metadata = state_dict._metadata 
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


def _named_modules_with_dup(
        model: nn.Module, prefix: str = ""
) -> Iterable[Tuple[str, nn.Module]]:
    yield prefix, model
    for name, module in model._modules.items():  
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)



def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth'))

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))