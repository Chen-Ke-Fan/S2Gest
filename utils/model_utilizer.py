import os
import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Any, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    SequentialLR,
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR
)


class SavePolicy(str, Enum):
    """Enumeration for supported model saving policies."""
    EARLY_STOP = "early_stop"
    ALL = "all"
    BEST = "best"


class OptimizerType(str, Enum):
    """Enumeration for supported optimizers."""
    ADAM = "Adam"
    ADAMW = "AdamW"
    RMSPROP = "RMSProp"
    SGD = "SGD"


class SchedulerType(str, Enum):
    """Enumeration for supported learning rate schedulers."""
    MULTI_STEP = "MultiStepLR"
    COSINE = "Cosine"
    COSINE_WARMUP = "CosineWarmup"


@dataclass(frozen=True)
class DefaultHyperparams:
    """
    Centralized default hyperparameter configuration.
    Using frozen=True ensures these values remain immutable (read-only) throughout training.
    """
    eta_min: float = 1e-6
    warmup_start_factor: float = 0.01
    gamma: float = 0.1
    warmup_epochs: int = 10
    decay_steps: Tuple[int, ...] = (100,)


# Global immutable hyperparameter defaults instance
DEFAULT_HP = DefaultHyperparams()



class ModuleUtilizer:
    """
    Model Utility Class.

    Responsibilities:
    1. Factory: Initialize optimizers and schedulers using Enum validation.
    2. Loader: Restore checkpoints or pretrained weights safely across devices.
    3. Checkpoint Manager: Manage metrics evaluation and file I/O operations cleanly
       via decoupled pipeline methods.
    """

    def __init__(self, configer: Any, logger: logging.Logger) -> None:
        """
        Args:
            configer: Configuration manager (must contain 'work_dir' and 'task_name').
            logger: Logging instance for output tracking.
        """
        self.configer = configer
        self.logger = logger
        self.device = self.configer.get("device")

        # 1. Distributed training and environment setup
        self.rank = self.configer.get("rank", default=0)
        self.is_main_process = (self.rank == 0)

        # 2. Path configuration
        work_dir = self.configer.get("work_dir")
        self.save_dir = Path(work_dir) if work_dir else Path("outputs")

        # 3. State trackers
        self.save_policy = self.configer.get("checkpoints", "save_policy")
        self.best_accuracy = 0.0
        self.last_improvement = 0  # Counter for early stopping


    def update_optimizer(self, net: nn.Module) -> Tuple[torch.optim.Optimizer, float]:
        """
        Creates and configures the optimizer based on Enum types.

        Args:
            net: The neural network model.

        Returns:
            A tuple containing the initialized optimizer and the base learning rate.
        """
        optim_type = self.configer.get('solver', 'type')
        base_lr = self.configer.get('solver', 'base_lr')
        weight_decay = self.configer.get('solver', 'weight_decay')

        # Retrieve model parameters (Compatible with DDP/DataParallel)
        if hasattr(net, "module"):
            params = net.module.parameters()
        else:
            params = net.parameters()

        # Filter out parameters that do not require gradients
        trainable_params = filter(lambda p: p.requires_grad, params)

        if self.is_main_process:
            self.logger.info(f"Optimizer: {optim_type} | Base LR: {base_lr} | Weight Decay: {weight_decay}")

        if optim_type == OptimizerType.ADAM:
            optimizer = torch.optim.Adam(trainable_params, lr=base_lr, weight_decay=weight_decay)
        elif optim_type == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)
        elif optim_type == OptimizerType.RMSPROP:
            optimizer = torch.optim.RMSprop(trainable_params, lr=base_lr, weight_decay=weight_decay)
        elif optim_type == OptimizerType.SGD:
            momentum = self.configer.get('solver', 'momentum')
            optimizer = torch.optim.SGD(trainable_params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError(
                f"Optimizer '{optim_type}' is not supported. Please choose from {list(OptimizerType)}."
            )

        return optimizer, base_lr

    def get_scheduler(self, optimizer: torch.optim.Optimizer, n_epochs: int) -> Any:
        """
        Factory method to create a learning rate scheduler using centralized default values.
        """
        policy = self.configer.get('solver', 'lr_policy', default=SchedulerType.MULTI_STEP)
        scheduler = None

        if policy == SchedulerType.MULTI_STEP:
            steps = self.configer.get('solver', 'decay_steps', default=list(DEFAULT_HP.decay_steps))
            gamma = self.configer.get('solver', 'gamma', default=DEFAULT_HP.gamma)
            scheduler = MultiStepLR(optimizer, milestones=steps, gamma=gamma)

        elif policy == SchedulerType.COSINE:
            eta_min = self.configer.get('solver', 'eta_min', default=DEFAULT_HP.eta_min)
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=eta_min)

        elif policy == SchedulerType.COSINE_WARMUP:
            warmup_epochs = self.configer.get('solver', 'warmup_epochs', default=DEFAULT_HP.warmup_epochs)
            eta_min = self.configer.get('solver', 'eta_min', default=DEFAULT_HP.eta_min)
            start_factor = self.configer.get('solver', 'warmup_start_factor', default=DEFAULT_HP.warmup_start_factor)

            sched_warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
            sched_cosine = CosineAnnealingLR(optimizer, T_max=(n_epochs - warmup_epochs), eta_min=eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])

        else:
            raise NotImplementedError(
                f"Scheduler policy '{policy}' is not supported. Please choose from {list(SchedulerType)}."
            )

        if self.is_main_process:
            self.logger.info(f"LR Scheduler Initialized: {policy}")

        return scheduler

    # ==========================================
    # Module 2: Model Loader (Resume & Pretrain)
    # ==========================================
    def load_net(self, net: nn.Module) -> Tuple[nn.Module, int, int, Optional[Dict[str, Any]]]:
        """
        Loads model weights (Handles both Resume Training and Pretrained Transfer Learning).
        """
        resume_path = self.configer.get('resume')
        pretrained_path = self.configer.get('network', 'pretrained', default=None)

        if resume_path is not None and os.path.exists(resume_path):
            if self.is_main_process:
                self.logger.info(f"Restoring checkpoint: {resume_path}")
            return self._load_checkpoint(net, resume_path)

        elif pretrained_path and os.path.exists(pretrained_path):
            if self.is_main_process:
                self.logger.info(f"Loading pretrained weights for new training: {pretrained_path}")
            net = self._load_pretrained_weights(net, pretrained_path)
            return net, 0, 0, None

        else:
            if self.is_main_process:
                self.logger.info("No resume or pretrained path provided. Training from scratch.")
            return net, 0, 0, None

    def _load_checkpoint(self, net: nn.Module, ckpt_path: str) -> Tuple[nn.Module, int, int, Dict[str, Any]]:
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v

        missing, unexpected = net.load_state_dict(new_state_dict, strict=True)
        if self.is_main_process:
            if missing: self.logger.warning(f"Missing keys during restoration: {missing}")
            if unexpected: self.logger.warning(f"Unexpected keys during restoration: {unexpected}")

        iters = checkpoint.get('iter', 0)
        epoch = checkpoint.get('epoch', 0)
        optimizer_state = checkpoint.get('optimizer', None)

        if 'best_acc' in checkpoint:
            self.best_accuracy = checkpoint['best_acc']
            if self.is_main_process:
                self.logger.info(f"Restored historical best accuracy: {self.best_accuracy:.4f}")

        net = net.to(self.device)
        return net, iters, epoch, optimizer_state

    def _load_pretrained_weights(self, net: nn.Module, weights_path: str) -> nn.Module:
        state_dict = torch.load(weights_path, map_location='cpu')

        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state_dict = {}
        current_model_dict = net.state_dict()

        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            if name in current_model_dict and v.shape == current_model_dict[name].shape:
                new_state_dict[name] = v
            else:
                if self.is_main_process:
                    self.logger.warning(f"Skipping mismatched pretrained weight: {name}")

        missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=False)
        if self.is_main_process:
            self.logger.info(f"Pretrained weights loaded. Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys found: {unexpected_keys}")

        net = net.to(self.device)
        return net


    def _evaluate_metrics(self, current_accuracy: float) -> Tuple[bool, bool]:
        """
        Pure state evaluation. Updates internal metrics and decides flags.
        """
        is_best = current_accuracy > self.best_accuracy
        should_stop = False

        if is_best:
            self.best_accuracy = current_accuracy
            self.last_improvement = 0  # Reset patience
        else:
            self.last_improvement += 1

        stop_patience = self.configer.get("checkpoints", "early_stop")
        # Support both the new Enum value and the legacy string variant 'earlystop' safely
        is_early_stop_policy = self.save_policy in [SavePolicy.EARLY_STOP, "earlystop"]

        if is_early_stop_policy and stop_patience:
            if self.last_improvement >= stop_patience:
                should_stop = True

        return is_best, should_stop

    def _unwrap_model_state(self, net: nn.Module) -> Dict[str, Any]:
        """
        Extracts pure state_dict, safely handling DDP wrappers.
        """
        if isinstance(net, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
            return net.module.state_dict()
        return net.state_dict()

    def _write_to_disk(
            self,
            state_dict: Dict[str, Any],
            optimizer: torch.optim.Optimizer,
            iters: int,
            epoch: int,
            filename: str
    ) -> None:
        """
        Pure file writing operation.
        """
        if not self.is_main_process:
            return

        state = {
            'iter': iters,
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
            'best_acc': self.best_accuracy
        }

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        file_path = self.save_dir / filename
        torch.save(state, file_path)

    def on_epoch_end(
            self,
            accuracy: float,
            net: nn.Module,
            optimizer: torch.optim.Optimizer,
            iters: int,
            epoch: int
    ) -> float:
        """
        Unified checkpoint pipeline called at the end of every epoch.

        Returns:
            -1.0 if Early Stopping is triggered, otherwise returns the historical best accuracy.
        """
        # 1. Evaluate Metrics (Logic)
        is_best, should_stop = self._evaluate_metrics(accuracy)

        # 2. Check Early Stopping Condition
        if should_stop:
            if self.is_main_process:
                self.logger.info(f"Early Stopping triggered. No improvement for {self.last_improvement} epochs.")
            return -1.0

            # 3. Execute I/O Pipeline (Only on Main Process)
        if self.is_main_process:
            task_name = self.configer.get("task_name", default="default_run")
            state_dict = self._unwrap_model_state(net)

            # Strategy: Save All Epochs
            if self.save_policy == SavePolicy.ALL:
                self._write_to_disk(state_dict, optimizer, iters, epoch, filename=f'{task_name}_epoch_{epoch}.pth')
                if is_best:
                    self._write_to_disk(state_dict, optimizer, iters, epoch, filename=f'best_{task_name}.pth')

            # Strategy: Save Best Only (or Default)
            else:
                if is_best:
                    self._write_to_disk(state_dict, optimizer, iters, epoch, filename=f'best_{task_name}.pth')

        return self.best_accuracy