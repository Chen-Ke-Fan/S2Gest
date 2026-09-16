<div align="center">

# S2Gest: Split-Scan State Space Models for Dynamic Hand Gesture Recognition

[![Conference](https://img.shields.io/badge/ECCV-2026-4b44ce.svg)](https://eccv2024.ecva.net/virtual/2026/poster/3811) [![Paper](https://img.shields.io/badge/Paper-Springer-orange.svg)](https://link.springer.com/chapter/10.1007/978-3-032-37447-9_27)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg)](https://pytorch.org/)

Official PyTorch implementation of the **ECCV 2026** paper "S2Gest: Split-Scan State Space Models for Dynamic Hand Gesture Recognition".

</div>

## 🌟 Introduction

This repository contains the official code for **S2Gest**. 

S2Gest explores the potential of State Space Models (specifically the Mamba architecture) for the task of Dynamic Hand Gesture Recognition. We propose a novel **Split-Scan** mechanism that effectively captures long-range spatio-temporal dependencies in gesture videos while maintaining an extremely low parameter count and high inference speed.

## 🛠️ Installation

> **⚠️ OS Requirement:** This project requires a **Linux** environment (e.g., Ubuntu) due to the strict dependencies of `mamba-ssm` and `triton`. If you are on Windows, we strongly advise using **WSL2** (Windows Subsystem for Linux).

Please ensure your environment meets the requirements. We recommend the following setup:
* **Python**: 3.12.3
* **PyTorch**: 2.2.1+cu118

**1. Create Environment & Install PyTorch**
```bash
conda create -n s2gest python=3.12.3 -y
conda activate s2gest
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

**2. Install Mamba-2 (Important)**
> **Note:** Direct compilation using `pip` often fails due to CUDA/C++ environment mismatch. We highly recommend downloading the pre-compiled `.whl` files from the official releases that match your PyTorch and Python versions.

```bash
pip install packaging ninja==1.13.0

# 1. Download causal-conv1d (v1.4.0) from: https://github.com/Dao-AILab/causal-conv1d/releases
# 2. Download mamba-ssm (v2.0.3) from: https://github.com/state-spaces/mamba/releases

# Install the downloaded wheels
pip install causal_conv1d-*.whl
pip install mamba_ssm-*.whl
```

**3. Install Other Requirements**


## 📂 Data Preparation

This project supports several mainstream dynamic hand gesture recognition datasets. Please download the datasets from their official sources using the links below:

**Supported Datasets:** [NVGesture](https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d) | [EgoGesture](https://nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html) | [Briareo](https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=31) | [Jester](https://20bn.com/datasets/jester)

Once downloaded, please organize your datasets following the directory structure below. *(Note: You can easily adapt this to your own custom paths by modifying the corresponding dataset paths in the `hyperparameters/` JSON files)*.

```text
data/
├── NVGesture/
│   ├── Video_data/
│   ├── nvgesture_train_correct_cvpr2016_v2.lst
│   └── ...
├── EgoGesture/
│   ├── images/
│   └── labels-final-revised1/
├── Briareo/
│   ├── test/
│   ├── train/
│   ├── validation/
│   ├── rgb_test.npz
│   └── ...
└── jester/
    ├── 20bn-jester-v1-videos/
    ├── jester-v1-train.csv
    ├── jester-v1-validation.csv
    └── ...
```


## 🚀 Usage

All training and evaluation tasks are launched from the `main.py` script. The core behavior is controlled by the `--phase` argument and a JSON configuration file specified with `--hypes`.

### Training

**1. Configure Your Experiment**

Before training, open a JSON configuration file in `hyperparameters/` (e.g., `hyperparameters/NVGesture/depth.json`). Adjust settings like `data_path`, `batch_size`, `epochs`, and learning rate (`lr`) as needed.

**2. Start Training**

*   **Single-GPU Training:**
    ```bash
    # Example: Train on NVGesture dataset using GPU 0 with the 'depth' config
    python main.py --phase train --hypes hyperparameters/NVGesture/depth.json --gpus 0
    ```

*   **Multi-GPU Training (Distributed):**
    We use `torchrun` for distributed training. The `--gpus` argument sets `CUDA_VISIBLE_DEVICES`.
    ```bash
    # Example: Train on 4 GPUs (0, 1, 2, 3)
    torchrun --nproc_per_node=4 main.py \
        --phase train \
        --hypes hyperparameters/NVGesture/depth.json \
        --gpus 0,1,2,3
    ```

**3. Resuming from a Checkpoint**

To continue training from a saved `.pth` file, use the `--resume` argument. The trainer will load the model, optimizer state, and last epoch number.
```bash
python main.py \
    --phase train \
    --hypes hyperparameters/NVGesture/depth.json \
    --resume ./outputs/NVGesture_depth/YOUR_TASK_NAME/model_best.pth \
    --gpus 0
```

### Evaluation

To evaluate a trained model, set `--phase test` and provide the model's configuration (`--hypes`) and weights (`--resume`). You can perform one or more evaluation tasks by appending the corresponding flags.

**General Command Format:**
```bash
python main.py \
    --phase test \
    --hypes path/to/config.json \
    --resume path/to/model.pth \
    --gpus 0 \
    [EVALUATION_FLAGS]
```

**Available Evaluation Flags:**

*   `--accuracy`: Computes Top-1 and Top-5 accuracy on the validation/test set.
*   `--matrix`: Use alongside `--accuracy` to also generate and save a confusion matrix plot.
*   `--benchmark`: Measures model efficiency (FLOPs, Parameters, Latency, FPS, and Throughput).
*   `--extract`: Extracts intermediate feature maps for a target sample for qualitative analysis.

**Examples:**

```bash
# Example 1: Run an accuracy test and also generate the confusion matrix
python main.py --phase test --hypes ... --resume ... --gpus 0 --accuracy --matrix

# Example 2: Run only the efficiency benchmark
python main.py --phase test --hypes ... --resume ... --gpus 0 --benchmark
```

## 📊 Model Zoo & Results

We provide pre-trained models for our S2Gest variants (Small, Tiny, Nano). You can download the specific weights by clicking on the corresponding accuracy score in the table. All results are reported as Top-1 Accuracy (%).

| Model | Params (M) | FLOPs (G) | Jester Pre-trained | NVGesture (RGB / Depth) | EgoGesture (RGB / Depth) | Briareo (RGB / Depth) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S2Gest-Small**| 2.59 | 14.90 | [Checkpoint](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/pretrained/s2gest_small_jester_rgb.pth) | [86.10](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_small_nvgesture_rgb_8610.pth) / [90.66](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_small_nvgesture_depth_9066.pth) | [96.47](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_small_egogesture_rgb_9647.pth) / [97.35](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_small_egogesture_depth_9735.pth) | [97.69](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_small_briareo_rgb_9769.pth) / [98.15](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_small_briareo_depth_9815.pth) |
| **S2Gest-Tiny** | 1.17 | 4.53 | [Checkpoint](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/pretrained/s2gest_tiny_jester_rgb.pth) | [85.06](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_tiny_nvgesture_rgb_8506.pth) / [87.76](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_tiny_nvgesture_depth_8776.pth) | [96.27](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_tiny_egogesture_rgb_9627.pth) / [97.27](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_tiny_egogesture_depth_9727.pth) | [97.69](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_tiny_briareo_rgb_9769.pth) / [98.61](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_tiny_briareo_depth_9861.pth) |
| **S2Gest-Nano** | 0.68 | 4.02 | [Checkpoint](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/pretrained/s2gest_nano_jester_rgb.pth) | [85.48](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_nano_nvgesture_rgb_8548.pth) / [82.99](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_nano_nvgesture_depth_8299.pth) | [95.94](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_nano_egogesture_rgb_9594.pth) / [97.07](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_nano_egogesture_depth_9707.pth) | [98.15](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_nano_briareo_rgb_9815.pth) / [97.69](https://github.com/Chen-Ke-Fan/S2Gest/raw/refs/heads/main/weights/s2gest_nano_briareo_depth_9769.pth) |


## 📜 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@InProceedings{S2Gest,
  author    = {Chen, Kefan and Gu, Yong and Li, Bo and Huang, Longjie and Zhang, Jiajun},
  title     = {S2Gest: Split-Scan State Space Models for Dynamic Hand Gesture Recognition},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026},
  pages     = {487--504},
  doi       = {10.1007/978-3-032-37447-9_27}
}
```
Chen, K., Gu, Y., Li, B., Huang, L., Zhang, J. (2026). S2Gest: Split-Scan State Space Models for Dynamic Hand Gesture Recognition. In *Computer Vision – ECCV 2026*. Lecture Notes in Computer Science, vol 17038. Springer, Cham. [https://doi.org/10.1007/978-3-032-37447-9_27](https://doi.org/10.1007/978-3-032-37447-9_27)

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
