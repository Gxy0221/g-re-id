import torch
import copy
import itertools
import math
import re
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from reid.config import CfgNode
from reid.utils.params import ContiguousParams
from . import lr_scheduler

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
        optimizer: Type[torch.optim.Optimizer],
        *,
        per_param_clipper: Optional[_GradientClipper] = None,
        global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    assert (
            per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    @torch.no_grad()
    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        optimizer.step(self, closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
        cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  
        return optimizer
    else:
        return OptimizerWithGradientClip


def _generate_optimizer_class_with_freeze_layer(
        optimizer: Type[torch.optim.Optimizer],
        *,
        freeze_iters: int = 0,
) -> Type[torch.optim.Optimizer]:
    assert freeze_iters > 0, "No layers need to be frozen or freeze iterations is 0"

    cnt = 0
    @torch.no_grad()
    def optimizer_wfl_step(self, closure=None):
        nonlocal cnt
        if cnt < freeze_iters:
            cnt += 1
            param_ref = []
            grad_ref = []
            for group in self.param_groups:
                if group["freeze_status"] == "freeze":
                    for p in group["params"]:
                        if p.grad is not None:
                            param_ref.append(p)
                            grad_ref.append(p.grad)
                            p.grad = None

            optimizer.step(self, closure)
            for p, g in zip(param_ref, grad_ref):
                p.grad = g
        else:
            optimizer.step(self, closure)

    OptimizerWithFreezeLayer = type(
        optimizer.__name__ + "WithFreezeLayer",
        (optimizer,),
        {"step": optimizer_wfl_step},
    )
    return OptimizerWithFreezeLayer


def maybe_add_freeze_layer(
        cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    if len(cfg.MODEL.FREEZE_LAYERS) == 0 or cfg.SOLVER.FREEZE_ITERS <= 0:
        return optimizer

    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    OptimizerWithFreezeLayer = _generate_optimizer_class_with_freeze_layer(
        optimizer_type,
        freeze_iters=cfg.SOLVER.FREEZE_ITERS
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithFreezeLayer  
        return optimizer
    else:
        return OptimizerWithFreezeLayer


def build_optimizer(cfg, model, contiguous=True):
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        heads_lr_factor=cfg.SOLVER.HEADS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        freeze_layers=cfg.MODEL.FREEZE_LAYERS if cfg.SOLVER.FREEZE_ITERS > 0 else [],
    )

    if contiguous:
        params = ContiguousParams(params)
    solver_opt = cfg.SOLVER.OPT
    if solver_opt == "SGD":
        return maybe_add_freeze_layer(
            cfg,
            maybe_add_gradient_clipping(cfg, torch.optim.SGD)
        )(
            params.contiguous() if contiguous else params,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
        ), params
    else:
        return maybe_add_freeze_layer(
            cfg,
            maybe_add_gradient_clipping(cfg, getattr(torch.optim, solver_opt))
        )(params.contiguous() if contiguous else params), params


def get_default_optimizer_params(
        model: torch.nn.Module,
        base_lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_norm: Optional[float] = None,
        bias_lr_factor: Optional[float] = 1.0,
        heads_lr_factor: Optional[float] = 1.0,
        weight_decay_bias: Optional[float] = None,
        overrides: Optional[Dict[str, Dict[str, float]]] = None,
        freeze_layers: Optional[list] = [],
):
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
      
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides

    layer_names_pattern = [re.compile(name) for name in freeze_layers]

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            hyperparams.update(overrides.get(module_param_name, {}))
            if module_name.split('.')[0] == "heads" and (heads_lr_factor is not None and heads_lr_factor != 1.0):
                hyperparams["lr"] = hyperparams.get("lr", base_lr) * heads_lr_factor
            name = module_name + '.' + module_param_name
            freeze_status = "normal"
            for pattern in layer_names_pattern:
                if pattern.match(name) is not None:
                    freeze_status = "freeze"
                    break

            params.append({"freeze_status": freeze_status, "params": [value], **hyperparams})
    return params


def build_lr_scheduler(cfg, optimizer, iters_per_epoch):
    max_epoch = cfg.SOLVER.MAX_EPOCH - max(
        math.ceil(cfg.SOLVER.WARMUP_ITERS / iters_per_epoch), cfg.SOLVER.DELAY_EPOCHS)

    scheduler_dict = {}

    scheduler_args = {
        "MultiStepLR": {
            "optimizer": optimizer,
            "milestones": cfg.SOLVER.STEPS,
            "gamma": cfg.SOLVER.GAMMA,
        },
        "CosineAnnealingLR": {
            "optimizer": optimizer,
            "T_max": max_epoch,
            "eta_min": cfg.SOLVER.ETA_MIN_LR,
        },

    }

    scheduler_dict["lr_sched"] = getattr(lr_scheduler, cfg.SOLVER.SCHED)(
        **scheduler_args[cfg.SOLVER.SCHED])

    if cfg.SOLVER.WARMUP_ITERS > 0:
        warmup_args = {
            "optimizer": optimizer,
            "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
            "warmup_method": cfg.SOLVER.WARMUP_METHOD,
        }
        scheduler_dict["warmup_sched"] = lr_scheduler.WarmupLR(**warmup_args)

    return scheduler_dict
