import os
import random
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import imgaug.augmenters as iaa

# Import our unified normalization tool
from datasets.data_utils.normalize import normalize


class Jester(Dataset):
    """
    20BN-Jester Dataset Loader (Production Grade)
    """

    def __init__(
            self,
            path: str,
            split: str = 'train',
            data_type: str = "color",
            transforms: Optional[iaa.Augmenter] = None,
            n_frames: int = 16,
            spatial_size: Tuple[int, int] = (320, 240)
    ):
        super().__init__()
        print(f"Loading Jester {split.upper()} dataset...", end=" ", flush=True)

        self.dataset_path = Path(path) / "Jester"
        self.split = split.lower()
        self.data_type = data_type.lower()
        self.spatial_size = spatial_size
        self.transforms = transforms
        self.n_frames = n_frames
        self.video_root = self.dataset_path / '20bn-jester-v1-videos'

        self.samples, self.label_map = self._build_or_load_cache()

    def _build_or_load_cache(self) -> Tuple[List[tuple], Dict[str, int]]:
        """
        Responsible for checking, loading, or building the high-speed Pickle cache from scratch.
        """
        cache_filename = f"jester-{self.split}-cache-nf{self.n_frames}.pkl"
        cache_path = self.dataset_path / cache_filename

        # 1. Cache hit: Lightning-fast loading
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                samples, label_map = pickle.load(f)
            print(f"done. Cache hit! Loaded {len(samples)} samples.")
            return samples, label_map

        # 2. Cache miss: Parse from scratch and persist
        print("\n[Init] No cache found. Parsing raw split files... (This happens only once)")
        split_name = self.split if self.split == "train" else "validation"
        split_file_path = self.dataset_path / f'jester-v1-{split_name}.csv'
        labels_file_path = self.dataset_path / 'jester-v1-labels.csv'

        # Pure function calls to avoid state pollution
        label_map = self._load_label_map(labels_file_path)
        samples = self._parse_split_file(split_file_path, label_map)

        # Write to cache
        with open(cache_path, 'wb') as f:
            pickle.dump((samples, label_map), f)

        print(f"Built and cached {len(samples)} samples.")
        return samples, label_map

    def _load_label_map(self, labels_file: Path) -> Dict[str, int]:
        """Parses the category mapping table."""
        labels_df = pd.read_csv(labels_file, header=None, names=['label_name'])
        return {name: idx for idx, name in enumerate(labels_df['label_name'])}

    def _parse_split_file(self, split_file: Path, label_map: Dict[str, int]) -> List[tuple]:
        """Stateless parser: Validates video paths and binds labels."""
        samples = []
        split_df = pd.read_csv(split_file, sep=';', header=None, names=['video_id', 'label_name'])

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Verifying Paths", leave=False):
            video_id, label_name = str(row['video_id']), row['label_name']
            video_path = self.video_root / video_id

            if video_path.exists():
                label_id = label_map[label_name]
                samples.append((video_id, label_id))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, num_total_frames: int) -> List[int]:
        """Precise temporal segment sampling."""
        if num_total_frames < self.n_frames:
            return np.linspace(0, num_total_frames - 1, self.n_frames).astype(int).tolist()

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

    def _load_frame(self, path: Path) -> np.ndarray:
        """Low-level Fail-Fast reading logic (Strictly prohibits returning None or black frames)."""
        if not path.exists():
            raise FileNotFoundError(f"Frame not found: {path}")

        with Image.open(path) as img:
            img_rgb = img.convert('RGB')
            # Unify physical spatial dimensions using the established contract
            # (Jester original frames vary in size; they must be aligned before stacking into a Numpy array)
            img_resized = img_rgb.resize(self.spatial_size)
            return np.array(img_resized, dtype=np.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            video_id, label_id = self.samples[index]
            video_path = self.video_root / video_id

            # 1. Get all image paths under the video directory
            frame_paths = sorted(list(video_path.glob('*.jpg')))
            num_total_frames = len(frame_paths)

            if num_total_frames == 0:
                raise RuntimeError(f"Video directory is empty: {video_path}")

            # 2. Calculate temporal indices (Core optimization: reject full loading, read only required frames)
            sampled_indices = self._sample_indices(num_total_frames)

            # 3. Precisely read the required frames (Triggers Fail-Fast)
            frames = []
            for idx in sampled_indices:
                img = self._load_frame(frame_paths[idx])
                frames.append(img)

            # 4. Assemble Tensor (H, W, C, T)
            data = np.stack(frames, axis=-1)

            # 5a. Normalize to bleach out illumination differences
            data = normalize(data, channel_axis=2)

            # 6a. Type Armor (float32): completely immunizes against imgaug float64 crash bugs
            # data = np.asarray(data, dtype=np.float32)

            # 7a. Synchronous spatial augmentation and dimension rearrangement
            if self.transforms is not None:
                aug_det = self.transforms.to_deterministic()
                data = np.array([
                    aug_det.augment_image(data[..., i]) for i in range(data.shape[-1])
                ]).transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            else:
                data = data.transpose(3, 2, 0, 1)  # (H, W, C, T) -> (T, C, H, W)

            # 8. Seamless PyTorch conversion
            data_tensor = torch.from_numpy(data).float().contiguous()
            label_tensor = torch.tensor(label_id, dtype=torch.long)

            return data_tensor, label_tensor

        except (RuntimeError, ValueError, FileNotFoundError, OSError) as e:
            # [Ultimate Fallback] Intercepts all I/O errors, logs the warning,
            # discards the dirty data, and resamples a healthy video.
            print(f"\n[Dataset Fallback] Skipping Jester sample {index}: {str(e)}")
            fallback_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(fallback_idx)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Add a robust transform pipeline to prove float64 immunity
    train_transforms = iaa.Sequential([
        iaa.Resize((0.8, 1.2)),
        iaa.CropToFixedSize(width=256, height=192),
        iaa.Rotate((-15, 15))
    ])

    dataset = Jester(
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
