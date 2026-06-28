import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from pathlib import Path
import imgaug.augmenters as iaa
from typing import Optional, Tuple
import random

# Import the strictly refactored architectural tools
from datasets.data_utils.normalize import normalize
from datasets.data_utils.utils_nvgesture import parse_nvgesture_split, load_video_frames


class NVGesture(Dataset):
    """
    NVGesture Dataset for PyTorch.

    A robust, fail-fast data loader providing spatio-temporal video tensors.
    Implements automatic corrupted-sample rejection and resampling mechanisms
    to maintain dataset integrity during large-scale training.
    """

    def __init__(
            self,
            path: str,
            split: str = "train",
            data_type: str = "depth",
            transforms: Optional[iaa.Augmenter] = None,
            n_frames: int = 40,
            spatial_size: Tuple[int, int] = (320, 240)
    ):
        super().__init__()

        print(f"Loading NVGesture {split.upper()} dataset...", end=" ", flush=True)

        self.dataset_path = Path(path) / "NVGesture"
        self.split = split
        self.transforms = transforms
        self.n_frames = n_frames
        self.spatial_size = spatial_size

        # Resolve dataset split list path
        split_suffix = "train" if self.split == "train" else "test"
        file_list_path = self.dataset_path / f"nvgesture_{split_suffix}_correct_cvpr2016_v2.lst"

        self.data_list = parse_nvgesture_split(split_file_path=file_list_path)

        # Domain Logic: Map data_type to sensor naming convention
        sensor_mapping = {
            "depth_z": "depth",
            "depth": "depth",
            "normal": "depth",
            "normals": "depth",
            "wrapped": "wrapped",
            "rgb": "color",
            "color": "color",
            "ir": "duo_left"
        }

        if data_type not in sensor_mapping:
            raise NotImplementedError(f"Modality '{data_type}' is not supported.")

        self.sensor = sensor_mapping[data_type]
        print(f"done. Loaded {len(self.data_list)} samples.")

    def __len__(self) -> int:
        return len(self.data_list)

    def _temporal_sampling(self, total_frames: int, n_samples: int) -> np.ndarray:
        """
        Generates temporally uniform indices for video frame sub-sampling.
        """
        if total_frames < n_samples:
            raise ValueError(
                f"Video duration ({total_frames} frames) is shorter than "
                f"requested sample size ({n_samples} frames)."
            )

        step = max(1, total_frames // n_samples)
        indices = np.arange(0, total_frames, step)[:n_samples]
        return indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a fully processed spatio-temporal tensor and its label.
        Includes a recursive fallback mechanism for I/O and duration errors.
        """
        try:
            sample_config = self.data_list[idx]

            # 1. I/O Loading (Returns shape: [H, W, C, T_actual])
            data, label = load_video_frames(
                data_root=self.dataset_path,
                sample_config=sample_config,
                sensor_type=self.sensor,
                target_size=self.spatial_size
            )

            # 2. Temporal Sampling
            total_frames = data.shape[-1]
            indices = self._temporal_sampling(total_frames, self.n_frames)
            data = data[..., indices]  # Shape: [H, W, C, n_frames]

            # [VERSION A: Normalize FIRST, then Augment]
            # ---------------------------------------------------------
            # 3a. Normalize (Layout is H, W, C, T -> Channel is at index 2)
            data = normalize(data, channel_axis=2)
            # data = np.asarray(data, dtype=np.float32)

            # 4a. Spatial Augmentation  (T, H, W, C) -> (T, C, H, W)
            if self.transforms is not None:
                aug_det = self.transforms.to_deterministic()
                data = np.array([
                    aug_det.augment_image(data[..., i]) for i in range(data.shape[-1])
                ]).transpose(0, 3, 1, 2)
            else:
                data = data.transpose(3, 2, 0, 1)
            # ---------------------------------------------------------

            # [VERSION B: Augment FIRST, then Normalize]
            # ---------------------------------------------------------
            # # 3b. Augment and Permute
            # if self.transforms is not None:
            #     aug_det = self.transforms.to_deterministic()
            #     data = np.array([
            #         aug_det.augment_image(data[..., i]) for i in range(data.shape[-1])
            #     ]).transpose(0, 3, 1, 2)  # Shape: (T, H, W, C) -> (T, C, H, W)
            # else:
            #     data = data.transpose(3, 2, 0, 1)  # Shape: (H, W, C, T) -> (T, C, H, W)
            # # 4b. Normalize (Layout is now T, C, H, W -> Channel is strictly at index 1)
            # data = normalize(data, channel_axis=1)
            # =========================================================

            # 5. PyTorch Conversion
            # `.contiguous()` is strictly required here because `.transpose`
            # breaks physical memory continuity, which degrades 3D Conv/Mamba performance.
            data_tensor = torch.from_numpy(data).float().contiguous()
            label_tensor = torch.tensor(label, dtype=torch.long)

            return data_tensor, label_tensor

        except (RuntimeError, ValueError) as e:
            # Defensive Fallback: If current video is corrupted or too short,
            # gracefully print the error and sample a random valid item.
            print(f"\n[Dataset Fallback] Skipping sample at index {idx}: {str(e)}")
            fallback_idx = random.randint(0, len(self.data_list) - 1)
            return self.__getitem__(fallback_idx)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 1. Define an aggressive augmentation pipeline to make the effects obvious
    transforms_pipeline = iaa.Sequential([
        iaa.Resize((0.8, 1.2)),
        iaa.CropToFixedSize(width=256, height=192),
        iaa.Rotate((-30, 30)),  # Increased rotation to see it clearly
        # iaa.Fliplr(0.5),
        # iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Add some noise

    ])

    # 2. Instantiate the dataset
    dataset = NVGesture(
        path="../data",
        split='train',
        data_type='rgb',
        n_frames=40,
        spatial_size=(320, 240),
        transforms=transforms_pipeline
    )

    # 3. Fetch the first processed video sample
    print("\nFetching sample and generating visualization...")
    sample_tensor, sample_label = dataset[0]
    print(f"Output Tensor Shape: {sample_tensor.shape}")
    print(f"Label: {sample_label}")

    # 4. Visualization Logic
    # Select 4 evenly spaced frames from the temporal dimension to visualize
    T = sample_tensor.shape[0]
    frame_indices = [0, T // 4, T // 2, 3 * T // 4]

    # Create a Matplotlib figure
    fig, axes = plt.subplots(1, len(frame_indices), figsize=(16, 4))
    fig.suptitle(f"Spatio-Temporal Augmentation (Label: {sample_label})", fontsize=16)

    for i, f_idx in enumerate(frame_indices):
        # Extract a single frame: (C, H, W) -> Permute to (H, W, C) for Matplotlib
        frame = sample_tensor[f_idx].permute(1, 2, 0).numpy()

        # Visual Denormalization:
        # The tensor is Z-score normalized (contains negative values).
        # We apply Min-Max scaling to map values back to [0.0, 1.0] strictly for viewing.
        frame_min = frame.min()
        frame_max = frame.max()
        frame_vis = (frame - frame_min) / (frame_max - frame_min + 1e-6)

        # Plot the frame
        axes[i].imshow(frame_vis)
        axes[i].set_title(f"Time Step: {f_idx}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
