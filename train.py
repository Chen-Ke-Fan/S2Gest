import sys
import numpy as np
import random
from typing import Any, Tuple
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import imgaug.augmenters as iaa
from tqdm import tqdm

from tensorboardX import SummaryWriter

# Import Datasets
from datasets.Briareo import Briareo
from datasets.EgoGesture import EgoGesture
from datasets.Jester import Jester
from datasets.NVGesture import NVGesture

# Import Model & Utils
from utils.model_utilizer import ModuleUtilizer
from utils.recorder import record_experiment
from utils.average_meter import AverageMeter

# Active Model Architecture
from models.build import build_model

LABEL_SMOOTHING_VAL = 0.1
MAX_GRAD_NORM = 1.0


def worker_init_fn(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader.
    Ensures that multiple processes have different random seeds for data augmentation
    in each epoch, preventing duplicated augmented samples.
    """
    np.random.seed(torch.initial_seed() % 2 ** 32)


class GestureTrainer:
    """
    Gesture Recognition Trainer.
    Supports Distributed Data Parallel (DDP) training and integrates TensorboardX for visualization.
    """

    def __init__(self, configer: Any, logger: logging.Logger) -> None:
        """
        Args:
            configer: Configuration manager (containing injected global paths like 'work_dir').
            logger: Logging instance for output tracking.
        """
        self.configer = configer
        self.logger = logger

        # Basic Device Configuration
        self.device = self.configer.get("device")
        self.data_path = self.configer.get("data", "data_path")

        # Distributed Configuration
        self.distributed = self.configer.get("distributed")
        self.rank = self.configer.get("rank", default=0)
        self.world_size = self.configer.get("world_size", default=1)
        self.local_rank = self.configer.get("local_rank", default=0)
        self.is_main_process = (self.rank == 0)

        # TensorBoard Initialization
        self.tbx_summary = None
        work_dir = self.configer.get("work_dir")

        if self.is_main_process and work_dir:
            self.tbx_summary = SummaryWriter(log_dir=str(work_dir))
            self.tbx_summary.add_text('Config', str(self.configer).replace("\n", "  \n"))
            if self.distributed:
                self.tbx_summary.add_text('Distributed', f'World Size: {self.world_size}')

        # Utility Modules
        self.model_utility = ModuleUtilizer(self.configer, self.logger)
        self.losses = {'train': AverageMeter(), 'val': AverageMeter()}
        self.accuracy = {'train': AverageMeter(), 'val': AverageMeter()}

        # Core Variables
        self.net = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.loss_fn = None
        self.optim_dict = None

        # Dataset Configuration
        self.dataset_name = self.configer.get("dataset").lower()
        self.data_type = self.configer.get("data", "type")
        self.clip_length = self.configer.get("data", "n_frames")
        self.n_classes = self.configer.get("data", "n_classes")
        self.optical_flow = self.configer.get("data", "optical_flow", default=False)
        self.total_epochs = self.configer.get("epochs")

        # Training States
        self.iters = 0
        self.epoch = 0
        self.in_planes = 3

    def init_model(self) -> None:
        """Master Entry: Initializes model, loss, optimizer, and dataloaders."""
        self._init_loss_function()
        self._build_model()
        self._wrap_distributed_model()
        self._init_optimizer_scheduler()
        self.init_dataloader()

    def _init_loss_function(self) -> None:
        """Decoupled: Initializes the loss function."""
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_VAL).to(self.device)

    def _build_model(self) -> None:
        """Decoupled: Constructs the network architecture."""
        if self.optical_flow:
            self.in_planes = 2
        elif self.data_type in ["depth", "ir", "depth_z"]:
            self.in_planes = 1
        else:
            self.in_planes = 3

        self.net = build_model(
            configer=self.configer,
            in_channels=self.in_planes,
            num_classes=self.n_classes
        )

        if self.is_main_process:
            model_class = self.net.__class__
            full_class_path = f"{model_class.__module__}.{model_class.__name__}"
            self.logger.info(f"Model Initialized: {full_class_path}")

        # Load Checkpoint / Pretrained Weights
        self.net, self.iters, self.epoch, self.optim_dict = self.model_utility.load_net(self.net)

        if self.epoch > 0:
            self.epoch += 1
            if self.is_main_process:
                self.logger.info(f"Resuming training from shifted start epoch: {self.epoch}")

        self.net = self.net.to(self.device)

    def _wrap_distributed_model(self) -> None:
        """Decoupled: Wraps the model with DDP and SyncBatchNorm if distributed."""
        if not self.distributed:
            return

        use_sync_bn = self.configer.get("sync_bn", default=False)
        if use_sync_bn:
            if self.is_main_process:
                self.logger.info("Converting BatchNorm to SyncBatchNorm...")
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)

        self.logger.info(f"Wrapping model with DDP on Local Rank: {self.local_rank}")
        self.net = DDP(
            self.net,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False
        )

    def _init_optimizer_scheduler(self) -> None:
        """Decoupled: Initializes optimizer and learning rate scheduler."""
        self.optimizer, self.lr = self.model_utility.update_optimizer(self.net)

        if self.optim_dict is not None:
            self.logger.info("Restoring optimizer state...")
            self.optimizer.load_state_dict(self.optim_dict)

        self.scheduler = self.model_utility.get_scheduler(self.optimizer, n_epochs=self.total_epochs)

        if self.epoch > 0:
            if self.is_main_process:
                self.logger.info(f"Fast-forwarding LR scheduler state to epoch {self.epoch}")
            for _ in range(self.epoch):
                self.scheduler.step()

    def _get_dataset_and_transforms(self) -> Tuple[Any, iaa.Sequential, iaa.Sequential]:
        """Helper function to construct dataset-specific transforms."""
        if self.dataset_name == "nvgesture":
            DatasetClass = NVGesture
            train_transforms = iaa.Sequential([
                iaa.Resize((0.8, 1.2)),
                iaa.CropToFixedSize(width=256, height=192),
                iaa.Rotate((-15, 15))
            ])
            val_transforms = iaa.CenterCropToFixedSize(width=256, height=192)

        elif self.dataset_name == "briareo":
            DatasetClass = Briareo
            train_transforms = iaa.Sequential([
                iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": 256}, interpolation="cubic"),
                iaa.CropToFixedSize(width=256, height=192),
                iaa.Rotate((-15, 15))
            ])
            val_transforms = iaa.Sequential([
                iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": 256}, interpolation="cubic"),
                iaa.CenterCropToFixedSize(width=256, height=192)
            ])

        elif self.dataset_name == "egogesture":
            DatasetClass = EgoGesture
            train_transforms = iaa.Sequential([
                iaa.Resize((0.8, 1.2)),
                iaa.CropToFixedSize(width=256, height=192),
                iaa.Rotate((-15, 15))
            ])
            val_transforms = iaa.CenterCropToFixedSize(width=256, height=192)

        elif self.dataset_name == "jester":
            DatasetClass = Jester
            train_transforms = iaa.Sequential([
                iaa.Resize((0.8, 1.2)),
                iaa.CropToFixedSize(width=256, height=192),
                iaa.Rotate((-15, 15))
            ])
            val_transforms = iaa.CenterCropToFixedSize(width=256, height=192)

        else:
            raise NotImplementedError(f"Dataset '{self.dataset_name}' is not supported.")

        return DatasetClass, train_transforms, val_transforms

    def init_dataloader(self) -> None:
        """Initializes the Datasets and DataLoaders."""
        DatasetClass, train_transforms, val_transforms = self._get_dataset_and_transforms()

        train_dataset = DatasetClass(
            self.data_path, split="train", data_type=self.data_type,
            transforms=train_transforms, n_frames=self.clip_length
        )

        val_dataset = DatasetClass(
            self.data_path, split="val", data_type=self.data_type,
            transforms=val_transforms, n_frames=self.clip_length
        )

        train_sampler = None
        val_sampler = None
        if self.distributed:
            seed = self.configer.get('seed', default=1994)
            if self.is_main_process:
                self.logger.info(f"DistributedSampler Seed: {seed}")

            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, drop_last=True, seed=seed
            )
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=False, drop_last=False, seed=seed
            )

        workers = self.configer.get('data', 'workers')
        batch_size = self.configer.get('data', 'batch_size')

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler,
            drop_last=True, num_workers=workers, pin_memory=True, worker_init_fn=worker_init_fn
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, sampler=val_sampler,
            drop_last=False, num_workers=workers, pin_memory=True, worker_init_fn=worker_init_fn
        )

        if self.is_main_process:
            self.logger.info(f"Data Loaded: Train Samples={len(train_dataset)}, Val Samples={len(val_dataset)}")

    def _train_epoch(self) -> None:
        """
        Decoupled: Executes one epoch of training.

        Expected Tensor Shapes:
            inputs: shape [B, C, T, H, W]
            labels: shape [B]
        """
        self.net.train()
        self.losses['train'].reset()
        self.accuracy['train'].reset()

        if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)

        iterator = self.train_loader
        if self.is_main_process:
            iterator = tqdm(iterator, desc=f"Train Epoch {self.epoch}", leave=False)

        for inputs, labels in iterator:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass
            output = self.net(inputs)
            loss = self.loss_fn(output, labels)

            # Backward pass & Gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            model_to_clip = self.net.module if hasattr(self.net, 'module') else self.net
            torch.nn.utils.clip_grad_norm_(model_to_clip.parameters(), max_norm=MAX_GRAD_NORM)
            self.optimizer.step()

            # Metrics Calculation
            with torch.no_grad():
                predicted = torch.argmax(output, dim=1)
                correct = (predicted == labels).float().mean()

            self.iters += 1
            self.losses['train'].update(loss.item(), inputs.size(0))
            self.accuracy['train'].update(correct.item(), inputs.size(0))

        # Update learning rate scheduler
        self.scheduler.step()

    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Decoupled: Model validation logic.

        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.net.eval()

        total_loss = torch.zeros(1, device=self.device)
        total_correct = torch.zeros(1, device=self.device)
        total_samples = torch.zeros(1, device=self.device)

        with torch.no_grad():
            iterator = self.val_loader
            if self.is_main_process:
                iterator = tqdm(iterator, desc="Val", leave=False)

            for inputs, labels in iterator:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                output = self.net(inputs)
                loss = self.loss_fn(output, labels)
                predicted = torch.argmax(output, dim=1)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (predicted == labels).sum()
                total_samples += inputs.size(0)

        if self.distributed:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        avg_loss = total_loss.item() / total_samples.item() if total_samples.item() > 0 else 0.0
        avg_acc = total_correct.item() / total_samples.item() if total_samples.item() > 0 else 0.0

        return avg_loss, avg_acc

    def _save_model(self, val_acc: float) -> float:
        """
        Decoupled: Checkpoint saving and early stopping evaluation.
        Interacts with the modernized ModuleUtilizer.
        """
        model_to_save = self.net.module if hasattr(self.net, "module") else self.net
        # Using the modernized facade method: on_epoch_end
        ret_code = self.model_utility.on_epoch_end(val_acc, model_to_save, self.optimizer, self.iters, self.epoch)
        return ret_code

    def _log_epoch_info(self, val_loss: float, val_acc: float, ret_code: float) -> None:
        """Decoupled: Terminal logging and TensorBoard recording."""
        current_lr = self.optimizer.param_groups[0]['lr']
        current_best = self.model_utility.best_accuracy
        epoch_width = len(str(self.total_epochs))

        # Status Suffix Determination
        status_suffix = ""
        if ret_code == -1.0:
            status_suffix = " | [Early Stop Triggered]"
        else:
            if val_acc >= current_best and val_acc > 0:
                status_suffix = " | [Saved Best]"
            elif self.model_utility.save_policy in ["all", "ALL"]:
                status_suffix = " | [Saved]"

        self.logger.info(
            f"Epoch {self.epoch + 1:>{epoch_width}}/{self.total_epochs} | "
            f"Train Loss: {self.losses['train'].avg:.4f} Acc: {self.accuracy['train'].avg:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} (Best: {current_best:.4f}) | "
            f"LR: {current_lr:.6f}"
            f"{status_suffix}"
        )

        if self.tbx_summary:
            self.tbx_summary.add_scalar('Train/Loss_Epoch', self.losses['train'].avg, self.epoch)
            self.tbx_summary.add_scalar('Train/Acc_Epoch', self.accuracy['train'].avg, self.epoch)
            self.tbx_summary.add_scalar('Val/Loss', val_loss, self.epoch)
            self.tbx_summary.add_scalar('Val/Acc', val_acc, self.epoch)

    def train(self) -> None:
        """Main Training Loop (Decoupled Pipeline)."""
        final_status = "Interrupted"

        try:
            for epoch in range(self.epoch, self.total_epochs):
                self.epoch = epoch

                # 1. Train
                self._train_epoch()

                # 2. Validate
                val_loss, val_acc = self._validate_epoch()

                # 3. Synchronize
                if self.distributed:
                    dist.barrier()

                should_stop = torch.tensor([0], dtype=torch.int32, device=self.device)

                # 4. Main Process I/O Pipeline
                if self.is_main_process:
                    ret_code = self._save_model(val_acc)
                    self._log_epoch_info(val_loss, val_acc, ret_code)

                    if ret_code == -1.0:
                        should_stop[0] = 1
                        final_status = "Early Stopped"

                # 5. Broadcast Stop Signal
                if self.distributed:
                    dist.broadcast(should_stop, src=0)

                if should_stop.item() == 1:
                    self.logger.info(f"Stopping training at Epoch {epoch}")
                    break

            if final_status == "Interrupted":
                final_status = "Finished"

        except KeyboardInterrupt:
            final_status = "User Cancelled"
            if self.is_main_process:
                self.logger.warning("\nKeyboardInterrupt detected. Saving logs and terminating...")
            raise

        except Exception as e:
            final_status = f"Error: {str(e)[:50]}"
            raise

        finally:
            if self.is_main_process:
                record_experiment(
                    file_path="experiment_history.csv",
                    task_name=self.configer.get("task_name"),
                    note=self.configer.get("note", default=""),
                    dataset=self.configer.get("dataset"),
                    modality=self.data_type,
                    best_acc=self.model_utility.best_accuracy,
                    status=final_status
                )
                if self.tbx_summary:
                    self.tbx_summary.close()
                self.logger.info(f"Training Ended. Status: {final_status}.")
