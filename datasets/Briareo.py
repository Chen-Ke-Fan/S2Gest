import os
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import imgaug.augmenters as iaa

# Import our unified tools
from datasets.data_utils.normalize import normalize
from datasets.data_utils.utils_briareo import parse_briareo_split, fix_briareo_path, load_briareo_frame


class Briareo(Dataset):
    """Briareo Dataset for PyTorch (Strict Architecture)."""

    def __init__(
            self,
            path: str,
            split: str = 'train',
            data_type: str = 'rgb',
            transforms: Optional[iaa.Augmenter] = None,
            n_frames: int = 40
    ):
        super().__init__()

        print(f"Loading Briareo {split.upper()} dataset...", end=" ", flush=True)

        self.dataset_path = Path(path) / "Briareo"
        self.split = split.lower()
        self.data_type = data_type.lower()
        self.n_frames = n_frames
        self.transforms = transforms

        file_list_path = self.dataset_path / f"{self.data_type}_{self.split}.npz"

        # Delegated to purely functional utility
        self.samples = parse_briareo_split(file_list_path)

        print(f"done. Loaded {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, num_total_frames: int) -> List[int]:
        if num_total_frames < self.n_frames:
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
            label = item['label']
            path_list = item['data']

            # 1. Temporal Sampling
            indices = self._sample_indices(len(path_list))

            # 2. Read Frames via Utility (Fail-Fast I/O)
            frames = []
            for i in indices:
                rel_path = path_list[i]
                full_path = os.path.join(self.dataset_path, fix_briareo_path(rel_path))
                img = load_briareo_frame(full_path, self.data_type)
                frames.append(img)

            # 3. Stack into (H, W, C, T)
            data = np.stack(frames, axis=-1)

            # 4. Pipeline Version A (Normalize -> Armor -> Augment)
            data = normalize(data, channel_axis=2)
            # data = np.asarray(data, dtype=np.float32)

            if self.transforms is not None:
                aug_det = self.transforms.to_deterministic()
                data = np.array([
                    aug_det.augment_image(data[..., i]) for i in range(data.shape[-1])
                ]).transpose(0, 3, 1, 2)
            else:
                data = data.transpose(3, 2, 0, 1)

            # 5. PyTorch Conversion
            data_tensor = torch.from_numpy(data).float().contiguous()
            label_tensor = torch.tensor(label, dtype=torch.long)

            return data_tensor, label_tensor

        except (RuntimeError, ValueError, FileNotFoundError, OSError) as e:
            print(f"\n[Dataset Fallback] Skipping sample at index {idx}: {str(e)}")
            fallback_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(fallback_idx)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 1. Define an aggressive augmentation pipeline to make the effects obvious
    transforms_pipeline = iaa.Sequential([
        iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": 256}, interpolation="cubic"),
        iaa.CropToFixedSize(width=256, height=192),
        iaa.Rotate((-15, 15)),
        # iaa.Fliplr(0.5),
        # iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Add some noise

    ])

    # 2. Instantiate the dataset
    dataset = Briareo(
        path="../data",
        split='train',
        data_type='depth',
        n_frames=40,
        transforms=transforms_pipeline
    )

    # 3. Fetch the first processed video sample
    print("\nFetching sample and generating visualization...")
    sample_tensor, sample_label = dataset[100]
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
