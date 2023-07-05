import logging
import time
import weakref
from typing import Dict
import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import reid.utils.comm as comm
from reid.utils.events import EventStorage, get_event_storage
from reid.utils.params import ContiguousParams

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]

logger = logging.getLogger(__name__)


class HookBase:

    def before_train(self):
        
        pass

    def after_train(self):
        
        pass

    def before_epoch(self):
        
        pass

    def after_epoch(self):
        
        pass

    def before_step(self):
       
        pass

    def after_step(self):
       
        pass


class TrainerBase:
  

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_epoch: int, max_epoch: int, iters_per_epoch: int):
        
        logger = logging.getLogger(__name__)
        logger.info("Starting training from epoch {}".format(start_epoch))

        self.iter = self.start_iter = start_epoch * iters_per_epoch

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.epoch in range(start_epoch, max_epoch):
                    self.before_epoch()
                    for _ in range(iters_per_epoch):
                        self.before_step()
                        self.run_step()
                        self.after_step()
                        self.iter += 1
                    self.after_epoch()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_epoch(self):
        self.storage.epoch = self.epoch

        for h in self._hooks:
            h.before_epoch()

    def before_step(self):
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    

    def __init__(self, model, data_loader, optimizer, param_wrapper):
       
        super().__init__()

       
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.param_wrapper = param_wrapper

    def run_step(self):
        
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

       
        self.optimizer.zero_grad()

        losses.backward()

        self._write_metrics(loss_dict, data_time)

        self.optimizer.step()
        if isinstance(self.param_wrapper, ContiguousParams):
            self.param_wrapper.assert_buffer_is_valid()

    def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float):

        device = next(iter(loss_dict.values())).device


        with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict["data_time"] = data_time


            all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()


            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)


            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)


class AMPTrainer(SimpleTrainer):
 

    def __init__(self, model, data_loader, optimizer, param_wrapper, grad_scaler=None):
        
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, param_wrapper)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
      
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # print(1111111111,data)

        with autocast():
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if isinstance(self.param_wrapper, ContiguousParams):
            self.param_wrapper.assert_buffer_is_valid()
