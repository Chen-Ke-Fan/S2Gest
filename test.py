import sys
import os
import argparse
import numpy as np
import time
from typing import Any, Tuple, List, Optional
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import imgaug.augmenters as iaa
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile, clever_format
import torch.backends.cudnn as cudnn

# Import Datasets
from datasets.Briareo import Briareo
from datasets.EgoGesture import EgoGesture
from datasets.Jester import Jester
from datasets.NVGesture import NVGesture

# Import Utility
from utils.model_utilizer import ModuleUtilizer
from utils.feature_extractor import FeatureExtractor

# Active Model Architecture
from models.build import build_model



def calculate_topk_counts(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    Calculates the absolute count of correct predictions for the specified top-k thresholds.

    Args:
        output: The predicted logits from the model (Shape: [B, num_classes]).
        target: The ground truth labels (Shape: [B]).
        topk: A tuple of integers representing the k values for top-k accuracy.

    Returns:
        A list of float values representing the raw count of correct predictions for each k.
    """
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item())

    return res


class GestureTester:
    """
    Gesture Recognition Tester.
    Handles accuracy evaluation, efficiency benchmarking, t-SNE visualization,
    and targeted feature extraction for analysis.
    """

    def __init__(self, configer: Any, logger: logging.Logger) -> None:
        self.configer = configer
        self.logger = logger
        self.device = torch.device(self.configer.get("device") if torch.cuda.is_available() else "cpu")
        self.work_dir = self.configer.get("work_dir")
        self.model_utility = ModuleUtilizer(self.configer, self.logger)

        # Dataset Parameters
        self.dataset_name = self.configer.get("dataset").lower()
        self.data_path = self.configer.get("data", "data_path")
        self.data_type = self.configer.get("data", "type")
        self.clip_length = self.configer.get("data", "n_frames")
        self.n_classes = self.configer.get("data", "n_classes")
        self.optical_flow = self.configer.get("data", "optical_flow", default=False)

        self.net = None
        self.val_loader = None

    def init_model(self) -> None:
        """
        Master Entry: Initializes model architecture and loads weights.
        Reuses ModuleUtilizer to ensure weight loading logic is consistent with training.
        """
        self._build_model()

        # Centralized loading logic (Handles DDP prefixes and strict matching automatically)
        self.net, _, _, _ = self.model_utility.load_net(self.net)

        self.net.eval()
        self.logger.info("Model initialized and weights loaded.")

    def _build_model(self) -> None:
        """Constructs the bare model architecture based on configurations."""
        if self.optical_flow:
            self.in_planes = 2
        elif self.data_type in ["depth", "ir", "depth_z"]:
            self.in_planes = 1
        else:
            self.in_planes = 3

        self.net = build_model(
            configer=self.configer,
            in_channels=self.in_planes,
            num_classes=self.n_classes,
            drop_path_rate=0.0,
            classifier_dropout=0.0
        )

        self.net = self.net.to(self.device)

        model_class = self.net.__class__
        full_class_path = f"{model_class.__module__}.{model_class.__name__}"
        self.logger.info(f"Model Initialized: {full_class_path}")

    def init_dataloader(self) -> None:
        """
        Initializes the validation DataLoader.
        Augmentation parameters are strictly localized per dataset to prevent coincidental cohesion.
        """
        if self.dataset_name == "nvgesture":
            DatasetClass = NVGesture
            val_transforms = iaa.CenterCropToFixedSize(width=256, height=192)
            split_to_use = "val"

        elif self.dataset_name == "egogesture":
            DatasetClass = EgoGesture
            val_transforms = iaa.CenterCropToFixedSize(width=256, height=192)
            split_to_use = "val"

        elif self.dataset_name == "briareo":
            DatasetClass = Briareo
            val_transforms = iaa.Sequential([
                iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": 256}, interpolation="cubic"),
                iaa.CenterCropToFixedSize(width=256, height=192)
            ])
            split_to_use = "val"

        elif self.dataset_name == "jester":
            DatasetClass = Jester
            val_transforms = iaa.CenterCropToFixedSize(width=256, height=192)
            split_to_use = "test"

        else:
            raise NotImplementedError(f"Dataset '{self.dataset_name}' is not supported.")

        val_dataset = DatasetClass(
            self.data_path,
            split=split_to_use,
            data_type=self.data_type,
            transforms=val_transforms,
            n_frames=self.clip_length
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.configer.get('data', 'batch_size'),
            shuffle=False,
            num_workers=self.configer.get('data', 'workers'),
            pin_memory=True
        )
        self.logger.info(f"Dataset Loaded: {self.dataset_name} ({split_to_use}) - {len(val_dataset)} samples")

    def run_accuracy_test(self, save_confusion_matrix: bool = False) -> None:
        """Executes full validation to compute Top-1 and Top-5 accuracy."""
        total_correct_1 = 0.0
        total_correct_5 = 0.0
        total_samples = 0.0
        all_preds = []
        all_labels = []

        self.logger.info("Starting Evaluation...")

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = self.net(inputs)
                correct_1, correct_5 = calculate_topk_counts(output, labels, topk=(1, 5))

                total_correct_1 += correct_1
                total_correct_5 += correct_5
                total_samples += inputs.size(0)

                if save_confusion_matrix:
                    _, pred = output.topk(1, 1, True, True)
                    all_preds.extend(pred.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())

        avg_top1 = (total_correct_1 / total_samples) * 100
        avg_top5 = (total_correct_5 / total_samples) * 100

        self.logger.info("\n" + "=" * 30)
        self.logger.info("🏆 Test Results:")
        self.logger.info(f"   Samples:   {int(total_samples)}")
        self.logger.info(f"   Top-1 Acc: {avg_top1:.2f}%")
        self.logger.info(f"   Top-5 Acc: {avg_top5:.2f}%")
        self.logger.info("=" * 30 + "\n")

        if save_confusion_matrix:
            self._plot_confusion_matrix(all_labels, all_preds)

    def _plot_confusion_matrix(self, labels: List[int], preds: List[int]) -> None:
        """Plots and saves the confusion matrix as a PNG image."""
        self.logger.info("Generating Confusion Matrix...")
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.dataset_name}')

        save_path = os.path.join(self.work_dir, f'confusion_matrix_{self.dataset_name}.png')
        plt.savefig(save_path)
        self.logger.info(f"Confusion Matrix saved to {save_path}")
        plt.close()

    def run_efficiency_benchmark(self) -> None:
        """Measures computational efficiency: Latency, FPS, Throughput, FLOPs, and Parameters."""
        self.logger.info("Starting Efficiency Benchmark...")

        h, w = (192, 256) if self.dataset_name in ["nvgesture", "egogesture", "briareo"] else (224, 224)
        input_shape = (1, self.clip_length, 1, h, w)
        dummy_input = torch.randn(input_shape).to(self.device)

        self._compute_flops_params(dummy_input)
        self._measure_latency_fps(dummy_input)
        self._measure_throughput(input_shape)

    def _compute_flops_params(self, dummy_input: torch.Tensor) -> None:
        """Calculates algorithmic complexity (FLOPs and Parameters)."""
        try:
            flops, params = profile(self.net, inputs=(dummy_input,), verbose=False)
            flops_str, params_str = clever_format([flops, params], "%.3f")
            self.logger.info("📏 Model Complexity:")
            self.logger.info(f"   Params: {params_str}")
            self.logger.info(f"   GFLOPs: {flops_str}")
        except Exception as e:
            self.logger.warning(f"FLOPs calculation failed: {e}")

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.logger.info(f"💻 Benchmarking on: {device_name}")

    def _measure_latency_fps(self, dummy_input: torch.Tensor) -> None:
        """Measures single-batch inference latency and FPS."""
        cudnn.benchmark = True
        n_warmup, n_runs = 50, 200

        self.logger.info("🔥 Warming up for Latency measurement...")
        with torch.no_grad():
            for _ in range(n_warmup):
                with torch.amp.autocast(device_type='cuda'):
                    _ = self.net(dummy_input)

        self.logger.info(f"⏱️ Measuring Latency ({n_runs} runs)...")
        timings = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(n_runs):
                start_event.record()
                with torch.amp.autocast(device_type='cuda'):
                    _ = self.net(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))

        avg_latency = np.mean(timings)
        fps = 1000 / avg_latency if avg_latency > 0 else 0

        self.logger.info("⚡ Latency Stats (BS=1):")
        self.logger.info(f"   Latency: {avg_latency:.2f} ms")
        self.logger.info(f"   FPS    : {fps:.2f}")

    def _measure_throughput(self, input_shape_single: Tuple[int, ...]) -> None:
        """Dynamically identifies max batch size and measures overall video processing throughput."""
        self.logger.info("\n🔍 Finding Max Batch Size for Throughput...")
        possible_bs = [1, 2, 4, 8, 16, 32]
        max_bs = 1

        for b in possible_bs:
            try:
                torch.cuda.empty_cache()
                x = torch.randn(b, *input_shape_single[1:]).to(self.device)
                with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                    _ = self.net(x)
                max_bs = b
                self.logger.info(f"  BS={b}: OK")
                del x
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.info(f"  BS={b}: OOM, max stable BS={max_bs}")
                    break
                else:
                    raise e

        n_warm = 5 if max_bs >= 16 else 20
        n_runs = 10 if max_bs >= 32 else 50

        self.logger.info(f"🚀 Measuring Throughput (BS={max_bs})")
        x = torch.randn(max_bs, *input_shape_single[1:]).to(self.device)

        with torch.no_grad():
            for _ in range(n_warm):
                with torch.amp.autocast(device_type='cuda'):
                    _ = self.net(x)
            torch.cuda.synchronize()

            start = time.time()
            for _ in tqdm(range(n_runs), desc="Benchmarking"):
                with torch.amp.autocast(device_type='cuda'):
                    _ = self.net(x)
                torch.cuda.synchronize()
            total_time = time.time() - start

        throughput = (max_bs * n_runs) / total_time if total_time > 0 else 0
        self.logger.info(f"🌊 Throughput: {throughput:.2f} videos/s")

        del x
        torch.cuda.empty_cache()

    def extract_target_sample_features(self, target_class_id: int = 0,
                                       target_layer_name: str = "mdse.cfd_proj.2",
                                       sample_index: int = 0) -> None:
        """
        Isolates a single valid sample of the specified class from the validation set
        and performs a forward pass to dynamically extract and save intermediate feature tensors.

        Args:
            target_class_id: The ID of the action class to visualize.
            target_layer_name: The exact string name of the layer to extract features from.
            sample_index: Which specific sample to pick across the ENTIRE dataset (0 = first, 1 = second...)
        """
        self.logger.info("=" * 40)
        self.logger.info(
            f"🔍 Searching validation set for sample index [{sample_index}] of target class [{target_class_id}]...")
        self.net.eval()
        found = False

        global_seen_count = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                if target_class_id not in labels:
                    continue

                matched_indices = (labels == target_class_id).nonzero(as_tuple=True)[0]
                num_in_batch = len(matched_indices)

                if global_seen_count + num_in_batch <= sample_index:
                    global_seen_count += num_in_batch
                    continue

                local_target_idx = sample_index - global_seen_count
                idx = matched_indices[local_target_idx]

                single_video = inputs[idx:idx + 1].to(self.device)
                single_label = labels[idx].item()

                self.logger.info(f"✅ Target sample located: Tensor shape {single_video.shape}")
                self.logger.info(f"🚀 Dynamically attaching hook to layer: [{target_layer_name}]")

                try:
                    with FeatureExtractor(self.net, target_layer_name) as extractor:
                        output = self.net(single_video)
                        extracted_feat = extractor.features

                except ValueError as e:
                    self.logger.error(e)
                    return

                save_filename = f"feat_class{target_class_id}_sample{sample_index}_{target_layer_name.replace('.', '_')}.npy"
                save_path = os.path.join(self.work_dir, save_filename)
                np.save(save_path, extracted_feat)

                self.logger.info(f"📦 Successfully extracted feature tensor of shape: {extracted_feat.shape}")
                self.logger.info(f"💾 Feature tensor saved safely to: {save_path}")

                pred_class = output.argmax(dim=1).item()
                self.logger.info(f"🎯 Ground Truth: {single_label} | Model Prediction: {pred_class}")

                if pred_class == single_label:
                    self.logger.info(
                        "✨ Classification matches Ground Truth. Sample is perfect for qualitative visualization.")
                else:
                    self.logger.warning(
                        "⚠️ Notice: The model misclassified this sample. Visualized feature maps may contain noise.")

                found = True
                break

        if not found:
            self.logger.error(
                f"❌ Could only find {global_seen_count} samples for class {target_class_id}. Sample index {sample_index} is out of bounds.")
        self.logger.info("=" * 40)
