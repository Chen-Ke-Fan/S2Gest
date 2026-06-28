import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Union


def parse_egogesture_labels(
        label_root: Union[str, Path],
        image_root: Union[str, Path],
        target_subjects: List[str],
        data_type: str
) -> List[Dict[str, Any]]:
    """
    Scans and parses EgoGesture CSV label files for the specified subjects.
    Constructs the absolute paths to the video frame directories.
    """
    label_root = Path(label_root)
    image_root = Path(image_root)
    samples = []

    for subj_id in target_subjects:
        label_subj_dir = label_root / subj_id.lower()
        if not label_subj_dir.exists():
            print(f"[Warning] EgoGesture label folder not found for {subj_id}")
            continue

        csv_files = glob.glob(os.path.join(label_subj_dir, "**", "*.csv"), recursive=True)

        for csv_path in csv_files:
            csv_path = Path(csv_path)
            parts = csv_path.parts

            group_filename = parts[-1]
            scene_name = parts[-2]
            group_id = group_filename.replace('Group', '').replace('.csv', '')

            if data_type == 'rgb':
                modality_folder = "Color"
                sub_folder = f"rgb{group_id}"
            elif data_type == 'depth':
                modality_folder = "Depth"
                sub_folder = f"depth{group_id}"
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

            video_path = image_root / subj_id / scene_name / modality_folder / sub_folder

            if not video_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path, header=None)
                for _, row in df.iterrows():
                    try:
                        label = int(row[0]) - 1  # 1-83 -> 0-82
                        start_f = int(row[1])
                        end_f = int(row[2])

                        samples.append({
                            'video_path': str(video_path),
                            'label': label,
                            'start': start_f,
                            'end': end_f,
                            'duration': end_f - start_f + 1
                        })
                    except ValueError:
                        continue
            except Exception as e:
                print(f"[Warning] Failed to parse CSV {csv_path}: {e}")

    return samples


def load_egogesture_frame(folder_path: str, frame_idx: int, data_type: str) -> np.ndarray:
    """
    Reads a single frame from the EgoGesture dataset.
    Strictly raises FileNotFoundError/OSError on failure to prevent silent data poisoning.
    """
    filename = f"{frame_idx:06d}.jpg"
    path = os.path.join(folder_path, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Frame not found: {path}")

    with Image.open(path) as img:
        if data_type == 'rgb':
            return np.array(img.convert('RGB'))
        elif data_type == 'depth':
            arr = np.array(img.convert('L'))
            return np.expand_dims(arr, axis=-1)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
