import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import imgaug.augmenters as iaa

# Import our unified tools
from datasets.data_utils.normalize import normalize
from datasets.data_utils.utils_egogesture import parse_egogesture_labels, load_egogesture_frame


class EgoGesture(Dataset):
    """
    EgoGesture Dataset for PyTorch.

    Implements strict fail-fast I/O operations and automatic corrupted-sample
    rejection mechanisms. Utilizes the robust 'Version A' preprocessing pipeline.
    """

    def __init__(
            self,
            path: str,
            split: str = "train",
            data_type: str = 'rgb',
            transforms: Optional[iaa.Augmenter] = None,
            n_frames: int = 40
    ):
        super().__init__()

        print(f"Loading EgoGesture {split.upper()} dataset...", end=" ", flush=True)

        self.dataset_path = Path(path) / "EgoGesture"
        self.label_root = self.dataset_path / "labels-final-revised1"
        self.image_root = self.dataset_path / "images"

        self.split = split.lower()
        self.data_type = data_type.lower()
        self.transforms = transforms
        self.n_frames = n_frames

        # 1. Subject Splitting Strategy
        if self.split == 'train':
            self.target_subjects = [f"Subject{i:02d}" for i in range(1, 46)]
        elif self.split == 'val':
            self.target_subjects = [f"Subject{i:02d}" for i in range(46, 51)]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        # 2. Delegate parsing to pure utility function
        self.samples = parse_egogesture_labels(
            self.label_root,
            self.image_root,
            self.target_subjects,
            self.data_type
        )

        print(f"done. Loaded {len(self.samples)} segments.")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, num_total_frames: int) -> List[int]:
        if num_total_frames <= 0:
            raise ValueError(f"Video too short: {num_total_frames} frames.")

        indices = []
        for i in range(self.n_frames):
            start_f = num_total_frames * i / self.n_frames
            end_f = num_total_frames * (i + 1) / self.n_frames
            start_idx, end_idx = int(start_f), int(end_f)

            if end_idx <= start_idx:
                end_idx = start_idx + 1
            end_idx = min(end_idx, num_total_frames)

            if self.split == 'train':
                idx = random.randint(start_idx, end_idx - 1) if start_idx < end_idx - 1 else start_idx
            else:
                idx = (start_idx + end_idx) // 2

            indices.append(min(idx, num_total_frames - 1))

        return indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            item = self.samples[idx]
            segment_len = item['duration']

            # 1. Temporal Sampling
            relative_indices = self._sample_indices(segment_len)

            # 2. Read Frames (Fail-Fast I/O)
            frames = []
            for rel_i in relative_indices:
                real_idx = item['start'] + rel_i
                img = load_egogesture_frame(item['video_path'], real_idx, self.data_type)
                frames.append(img)

            # 3. Stack into (H, W, C, T)
            data = np.stack(frames, axis=-1)

            # 4a. Normalize
            data = normalize(data, channel_axis=2)
            # data = np.asarray(data, dtype=np.float32)

            # 5a. Spatial Augmentation & Permutation
            if self.transforms is not None:
                aug_det = self.transforms.to_deterministic()

                data = np.array([
                    aug_det.augment_image(data[..., i]) for i in range(data.shape[-1])
                ]).transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            else:
                data = data.transpose(3, 2, 0, 1)  # (H, W, C, T) -> (T, C, H, W)

            # 6. PyTorch Conversion
            data_tensor = torch.from_numpy(data).float().contiguous()
            label_tensor = torch.tensor(item['label'], dtype=torch.long)

            return data_tensor, label_tensor

        except (RuntimeError, ValueError, FileNotFoundError, OSError) as e:
            # Defensive Fallback: Automatically sample another segment if I/O fails
            print(f"\n[Dataset Fallback] Skipping sample at index {idx}: {str(e)}")
            fallback_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(fallback_idx)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Add a robust transform pipeline to prove float64 immunity
    train_transforms = iaa.Sequential([
        iaa.Resize((0.8, 1.2)),
        iaa.CropToFixedSize(width=256, height=192),
        iaa.Rotate((-15, 15))
    ])

    dataset = EgoGesture(
        path="../data",
        split="train",
        data_type='rgb',
        n_frames=40,
        transforms=train_transforms
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
