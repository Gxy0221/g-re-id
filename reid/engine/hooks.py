import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import datetime
import itertools
import logging
import os
import tempfile
import time
from collections import Counter
from reid.evaluation.testing import flatten_results_dict
from reid.utils import comm
from reid.utils.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from reid.utils.events import EventStorage, EventWriter, get_event_storage
from reid.utils.file_io import PathManager
from reid.utils.precision_bn import update_bn_stats, get_bn_modules
from reid.utils.timer import Timer
from .train_loop import HookBase

__all__ = [
    "CallbackHook",
    "IterationTimer",
    "PeriodicWriter",
    "PeriodicCheckpointer",
    "LRScheduler",
    "AutogradProfiler",
    "EvalHook",
    "PreciseBN",
    "LayerFreeze",
]

class CallbackHook(HookBase):
    def __init__(self, *, before_train=None, after_train=None, before_epoch=None, after_epoch=None,
                 before_step=None, after_step=None):
        self._before_train = before_train
        self._before_epoch = before_epoch
        self._before_step = before_step
        self._after_step = after_step
        self._after_epoch = after_epoch
        self._after_train = after_train

    def before_train(self):
        if self._before_train:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:
            self._after_train(self.trainer)
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_epoch(self):
        if self._before_epoch:
            self._before_epoch(self.trainer)

    def after_epoch(self):
        if self._after_epoch:
            self._after_epoch(self.trainer)

    def before_step(self):
        if self._before_step:
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:
            self._after_step(self.trainer)


class IterationTimer(HookBase):
    def __init__(self, warmup_iter=3):
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer = Timer()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )
        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


class PeriodicWriter(HookBase):
    def __init__(self, writers, period=20):
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (
                self.trainer.iter == self.trainer.max_iter - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_epoch(self):
        for writer in self._writers:
            writer.write()

    def after_train(self):
        for writer in self._writers:
            writer.close()


class PeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    def before_train(self):
        self.max_epoch = self.trainer.max_epoch
        if len(self.trainer.cfg.DATASETS.TESTS) == 1:
            self.metric_name = "metric"
        else:
            self.metric_name = self.trainer.cfg.DATASETS.TESTS[0] + "/metric"

    def after_epoch(self):
        storage = get_event_storage()
        metric_dict = dict(
            metric=storage.latest()[self.metric_name][0] if self.metric_name in storage.latest() else -1
        )
        self.step(self.trainer.epoch, **metric_dict)


class LRScheduler(HookBase):
    def __init__(self, optimizer, scheduler):
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scale = 0
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def before_step(self):
        if self.trainer.grad_scaler is not None:
            self._scale = self.trainer.grad_scaler.get_scale()

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)

        next_iter = self.trainer.iter + 1
        if next_iter <= self.trainer.warmup_iters:
            if self.trainer.grad_scaler is None or self._scale == self.trainer.grad_scaler.get_scale():
                self._scheduler["warmup_sched"].step()

    def after_epoch(self):
        next_iter = self.trainer.iter + 1
        next_epoch = self.trainer.epoch + 1
        if next_iter > self.trainer.warmup_iters and next_epoch > self.trainer.delay_epochs:
            self._scheduler["lr_sched"].step()


class AutogradProfiler(HookBase):
    def __init__(self, enable_predicate, output_dir, *, use_cuda=True):
        self._enable_predicate = enable_predicate
        self._use_cuda = use_cuda
        self._output_dir = output_dir

    def before_step(self):
        if self._enable_predicate(self.trainer):
            self._profiler = torch.autograd.profiler.profile(use_cuda=self._use_cuda)
            self._profiler.__enter__()
        else:
            self._profiler = None

    def after_step(self):
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        out_file = os.path.join(
            self._output_dir, "profiler-trace-iter{}.json".format(self.trainer.iter)
        )
        if "://" not in out_file:
            self._profiler.export_chrome_trace(out_file)
        else:
            with tempfile.TemporaryDirectory(prefix="reid_profiler") as d:
                tmp_file = os.path.join(d, "tmp.json")
                self._profiler.export_chrome_trace(tmp_file)
                with open(tmp_file) as f:
                    content = f.read()
            with PathManager.open(out_file, "w") as f:
                f.write(content)


class EvalHook(HookBase):
    def __init__(self, eval_period, eval_function):
        self._period = eval_period
        self._func = eval_function

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    )
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        torch.cuda.empty_cache()
        comm.synchronize()

    def after_epoch(self):
        next_epoch = self.trainer.epoch + 1
        if self._period > 0 and next_epoch % self._period == 0:
            self._do_eval()

    def after_train(self):
        next_epoch = self.trainer.epoch + 1     
        if next_epoch % self._period != 0 and next_epoch >= self.trainer.max_epoch:
            self._do_eval()       
        del self._func


class PreciseBN(HookBase):
    def __init__(self, model, data_loader, num_iter):     
        self._logger = logging.getLogger(__name__)
        if len(get_bn_modules(model)) == 0:
            self._logger.info(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
            )
            self._disabled = True
            return

        self._model = model
        self._data_loader = data_loader
        self._num_iter = num_iter
        self._disabled = False

        self._data_iter = None

    def after_epoch(self):
        next_epoch = self.trainer.epoch + 1
        is_final = next_epoch == self.trainer.max_epoch
        if is_final:
            self.update_stats()

    def update_stats(self):        
        if self._disabled:
            return

        if self._data_iter is None:
            self._data_iter = iter(self._data_loader)

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    self._logger.info(
                        "Running precise-BN ... {}/{} iterations.".format(num_iter, self._num_iter)
                    )
               
                yield next(self._data_iter)

        with EventStorage():  
            self._logger.info(
                "Running precise-BN for {} iterations...  ".format(self._num_iter)
                + "Note that this could produce different statistics every time."
            )
            update_bn_stats(self._model, data_loader(), self._num_iter)


class LayerFreeze(HookBase):
    def __init__(self, model, freeze_layers, freeze_iters):
        self._logger = logging.getLogger(__name__)
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.model = model

        self.freeze_layers = freeze_layers
        self.freeze_iters = freeze_iters

        self.is_frozen = False

    def before_step(self):        
        if self.trainer.iter < self.freeze_iters and not self.is_frozen:
            self.freeze_specific_layer()       
        if self.trainer.iter >= self.freeze_iters and self.is_frozen:
            self.open_all_layer()

    def freeze_specific_layer(self):
        for layer in self.freeze_layers:
            if not hasattr(self.model, layer):
                self._logger.info(f'{layer} is not an attribute of the model, will skip this layer')

        for name, module in self.model.named_children():
            if name in self.freeze_layers:
                module.eval()

        self.is_frozen = True
        freeze_layers = ", ".join(self.freeze_layers)
        self._logger.info(f'Freeze layer group "{freeze_layers}" training for {self.freeze_iters:d} iterations')

    def open_all_layer(self):
        for name, module in self.model.named_children():
            if name in self.freeze_layers:
                module.train()

        self.is_frozen = False

        freeze_layers = ", ".join(self.freeze_layers)
        self._logger.info(f'Open layer group "{freeze_layers}" training')


class SWA(HookBase):
    def __init__(self, swa_start: int, swa_freq: int, swa_lr_factor: float, eta_min: float, lr_sched=False, ):
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr_factor = swa_lr_factor
        self.eta_min = eta_min
        self.lr_sched = lr_sched

    def before_step(self):
        is_swa = self.trainer.iter == self.swa_start
        if is_swa:
            self.trainer.optimizer = optim.SWA(self.trainer.optimizer, self.swa_freq, self.swa_lr_factor)
            self.trainer.optimizer.reset_lr_to_swa()

            if self.lr_sched:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=self.trainer.optimizer,
                    T_0=self.swa_freq,
                    eta_min=self.eta_min,
                )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter > self.swa_start and self.lr_sched:
            self.scheduler.step()

        is_final = next_iter == self.trainer.max_iter
        if is_final:
            self.trainer.optimizer.swap_swa_param()
