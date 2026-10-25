import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Union


def parse_briareo_split(split_file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Parses the Briareo .npz split file and filters out invalid sequences (e.g., g12).
    """
    file_path = Path(split_file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    try:
        raw_data = np.load(str(file_path), allow_pickle=True)['arr_0']
    except Exception as e:
        raise RuntimeError(f"Failed to load Briareo split file {file_path}: {e}")

    valid_samples = []
    for item in raw_data:
        if int(item['label']) == 12:  # Filter out g12 (long videos)
            continue
        valid_samples.append(item)

    return valid_samples


def fix_briareo_path(path_in_npz: str) -> str:
    """Standardizes path separators across different OS environments."""
    parts = path_in_npz.replace('\\', '/').split('/')
    if parts[0] in ['rgb', 'tof', 'depth', 'ir']:
        return os.path.join(*parts[1:])
    return path_in_npz


def load_briareo_frame(file_path: Union[str, Path], data_type: str) -> np.ndarray:
    """
    Reads a single frame from disk and returns a Numpy array (H, W, C).
    Strictly raises an exception on failure to ensure Fail-Fast I/O.
    """
    path_str = str(file_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Frame file not found: {path_str}")

    if data_type == 'rgb':
        with Image.open(path_str) as img:
            return np.array(img.convert('RGB'))  # Shape: (H, W, 3)

    elif data_type == 'depth':
        arr = np.load(path_str)['arr_0'].astype(np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)  # Shape: (H, W, 1)
        return arr

    else:
        raise ValueError(f"Unknown modality: {data_type}")


# full 675, full_no_fingers 145, mod 192
def from_json_to_list(json_file):
    with open(json_file) as f:
        j = json.load(f)
        if j['frame'] != 'invalid':
            j_vector = [
                # palm
                j['frame']['right_hand']['palm_position'][0],
                j['frame']['right_hand']['palm_position'][1],
                j['frame']['right_hand']['palm_position'][2],
                j['frame']['right_hand']['palm_position'][3],
                j['frame']['right_hand']['palm_position'][4],
                j['frame']['right_hand']['palm_position'][5],
                j['frame']['right_hand']['palm_normal'][0],
                j['frame']['right_hand']['palm_normal'][1],
                j['frame']['right_hand']['palm_normal'][2],
                j['frame']['right_hand']['palm_normal'][3],
                j['frame']['right_hand']['palm_normal'][4],
                j['frame']['right_hand']['palm_normal'][5],
                j['frame']['right_hand']['palm_velocity'][0],
                j['frame']['right_hand']['palm_velocity'][1],
                j['frame']['right_hand']['palm_velocity'][2],
                j['frame']['right_hand']['palm_velocity'][3],
                j['frame']['right_hand']['palm_velocity'][4],
                j['frame']['right_hand']['palm_velocity'][5],
                j['frame']['right_hand']['palm_width'],
                j['frame']['right_hand']['pinch_strength'],
                j['frame']['right_hand']['grab_strength'],
                j['frame']['right_hand']['direction'][0],
                j['frame']['right_hand']['direction'][1],
                j['frame']['right_hand']['direction'][2],
                j['frame']['right_hand']['direction'][3],
                j['frame']['right_hand']['direction'][4],
                j['frame']['right_hand']['direction'][5],
                j['frame']['right_hand']['sphere_center'][0],
                j['frame']['right_hand']['sphere_center'][1],
                j['frame']['right_hand']['sphere_center'][2],
                j['frame']['right_hand']['sphere_center'][3],
                j['frame']['right_hand']['sphere_center'][4],
                j['frame']['right_hand']['sphere_center'][5],
                j['frame']['right_hand']['sphere_radius'],
                # wrist
                j['frame']['right_hand']['wrist_position'][0],
                j['frame']['right_hand']['wrist_position'][1],
                j['frame']['right_hand']['wrist_position'][2],
                j['frame']['right_hand']['wrist_position'][3],
                j['frame']['right_hand']['wrist_position'][4],
                j['frame']['right_hand']['wrist_position'][5],
                # pointables
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_0']['direction'][0],
                j['frame']['right_hand']['pointables']['p_0']['direction'][1],
                j['frame']['right_hand']['pointables']['p_0']['direction'][2],
                j['frame']['right_hand']['pointables']['p_0']['direction'][3],
                j['frame']['right_hand']['pointables']['p_0']['direction'][4],
                j['frame']['right_hand']['pointables']['p_0']['direction'][5],
                j['frame']['right_hand']['pointables']['p_0']['width'],
                j['frame']['right_hand']['pointables']['p_0']['length'],
                float(j['frame']['right_hand']['pointables']['p_0']['is_extended']),
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_1']['direction'][0],
                j['frame']['right_hand']['pointables']['p_1']['direction'][1],
                j['frame']['right_hand']['pointables']['p_1']['direction'][2],
                j['frame']['right_hand']['pointables']['p_1']['direction'][3],
                j['frame']['right_hand']['pointables']['p_1']['direction'][4],
                j['frame']['right_hand']['pointables']['p_1']['direction'][5],
                j['frame']['right_hand']['pointables']['p_1']['width'],
                j['frame']['right_hand']['pointables']['p_1']['length'],
                float(j['frame']['right_hand']['pointables']['p_1']['is_extended']),
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_2']['direction'][0],
                j['frame']['right_hand']['pointables']['p_2']['direction'][1],
                j['frame']['right_hand']['pointables']['p_2']['direction'][2],
                j['frame']['right_hand']['pointables']['p_2']['direction'][3],
                j['frame']['right_hand']['pointables']['p_2']['direction'][4],
                j['frame']['right_hand']['pointables']['p_2']['direction'][5],
                j['frame']['right_hand']['pointables']['p_2']['width'],
                j['frame']['right_hand']['pointables']['p_2']['length'],
                float(j['frame']['right_hand']['pointables']['p_2']['is_extended']),
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_3']['direction'][0],
                j['frame']['right_hand']['pointables']['p_3']['direction'][1],
                j['frame']['right_hand']['pointables']['p_3']['direction'][2],
                j['frame']['right_hand']['pointables']['p_3']['direction'][3],
                j['frame']['right_hand']['pointables']['p_3']['direction'][4],
                j['frame']['right_hand']['pointables']['p_3']['direction'][5],
                j['frame']['right_hand']['pointables']['p_3']['width'],
                j['frame']['right_hand']['pointables']['p_3']['length'],
                float(j['frame']['right_hand']['pointables']['p_3']['is_extended']),
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_4']['direction'][0],
                j['frame']['right_hand']['pointables']['p_4']['direction'][1],
                j['frame']['right_hand']['pointables']['p_4']['direction'][2],
                j['frame']['right_hand']['pointables']['p_4']['direction'][3],
                j['frame']['right_hand']['pointables']['p_4']['direction'][4],
                j['frame']['right_hand']['pointables']['p_4']['direction'][5],
                j['frame']['right_hand']['pointables']['p_4']['width'],
                j['frame']['right_hand']['pointables']['p_4']['length'],
                float(j['frame']['right_hand']['pointables']['p_4']['is_extended']),
            ]
        else:
            j_vector = False

        return j_vector, j
