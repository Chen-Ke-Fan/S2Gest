
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union


def parse_nvgesture_split(
        split_file_path: Union[str, Path]
) -> List[Dict[str, Any]]:
    """
    Parses the NVGesture dataset split (.lst) file to extract metadata and paths.

    Args:
        split_file_path (Union[str, Path]): Path to the split list file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the configuration
        for each data sample.

    Raises:
        FileNotFoundError: If the split file does not exist.
    """
    file_path = Path(split_file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    # Dynamically extract dataset name prefix (e.g., 'nvgesture' from 'nvgesture_train.lst')
    dict_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem

    parsed_configs: List[Dict[str, Any]] = []

    with file_path.open('rt') as f:
        for line in f:
            params = line.strip().split(' ')
            if not params:
                continue

            config: Dict[str, Any] = {'dataset': dict_name}

            # params[0] format -> "ClassID:path/to/video"
            base_path = params[0].split(':')[1]

            for param in params[1:]:
                parsed = param.split(':')
                key = parsed[0]

                if key == 'label':
                    # NVGesture labels are 1-indexed in the file; convert to 0-indexed for training
                    config['label'] = int(parsed[1]) - 1
                elif key in ('depth', 'color', 'duo_left'):
                    config[key] = f"{base_path}/{parsed[1]}"
                    config[f"{key}_start"] = int(parsed[2])
                    config[f"{key}_end"] = int(parsed[3])

            # Auto-complete stereo/disparity modalities based on 'duo_left'
            if 'duo_left' in config:
                for target in ('duo_right', 'duo_disparity'):
                    config[target] = config['duo_left'].replace('duo_left', target)
                    config[f"{target}_start"] = config['duo_left_start']
                    config[f"{target}_end"] = config['duo_left_end']

            parsed_configs.append(config)

    return parsed_configs


def load_video_frames(
        data_root: Union[str, Path],
        sample_config: Dict[str, Any],
        sensor_type: str,
        target_size: Tuple[int, int],
        expected_frame_count: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Loads and preprocesses a specific segment of a video file based on the config.

    Args:
        data_root (Union[str, Path]): Root directory containing the video files.
        sample_config (Dict[str, Any]): Metadata dictionary for a single sample.
        sensor_type (str): The modality to load (e.g., 'color', 'depth', 'duo_left').
        target_size (Tuple[int, int]): Target spatial resolution as (width, height).
        expected_frame_count (Optional[int]): Expected temporal length for validation.

    Returns:
        Tuple[np.ndarray, int]: The structured video tensor (H, W, C, T) and its label.

    Raises:
        RuntimeError: If video parsing fails to prevent silent data poisoning.
    """
    root_path = Path(data_root)
    video_path = root_path / f"{sample_config[sensor_type]}.avi"

    start_frame = sample_config[f"{sensor_type}_start"]
    end_frame = sample_config[f"{sensor_type}_end"]
    label = sample_config['label']

    target_frames = end_frame - start_frame

    # Native print warning as requested
    if expected_frame_count is not None and target_frames != expected_frame_count:
        print(f"[WARNING] Video {video_path.name} has {target_frames} frames (Expected {expected_frame_count}).")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Critical I/O Error: Could not open video file {video_path}")
        raise RuntimeError(f"Failed to open {video_path}")

    target_width, target_height = target_size
    channels = 3 if sensor_type == "color" else 1

    video_container = np.zeros(
        (target_height, target_width, channels, target_frames),
        dtype=np.uint8
    )

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(target_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Unexpected EOF at frame {start_frame + i} in {video_path}")
                raise RuntimeError(f"Corrupted frame data in {video_path}")

            frame = cv2.resize(frame, (target_width, target_height))

            if sensor_type == "color":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = frame[:, :, 0:1]

            video_container[..., i] = frame

    finally:
        cap.release()

    # [H, W, C, T_actual]
    return video_container, label
