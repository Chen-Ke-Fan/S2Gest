import argparse
import datetime
import random
import time
import os
import shutil
from typing import Any, Tuple
import logging

import numpy as np
import torch
import torch.distributed as dist

from utils.configer import Configer
from utils.logger import setup_logger
from train import GestureTrainer
from test import GestureTester

# =====================================================================
# Global Constants (Immutable configurations)
# =====================================================================
DEFAULT_CONFIG_PATH = "hyperparameters/NVGesture/depth.json"
DEFAULT_SEED = 1994
DEFAULT_MODALITY = "DefaultModality"
DEFAULT_DATASET = "DefaultDataset"
DEFAULT_PROJECT_NAME = "GestureRecognition"
DEFAULT_OUTPUT_DIR = "./outputs"
REBUTTAL_TARGET_ACTION_ID = 8

BACKUP_SOURCE_DIRS: Tuple[str, ...] = (
    'main.py',
    'train.py',
    'test.py',
    'datasets',
    'hyperparameters',
    'models',
    'tools',
    'utils'
)


# =====================================================================
# Pipeline Helper Functions
# =====================================================================

def set_seed(seed: int) -> None:
    """Sets the global random seed to ensure experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _generate_timestamp_name(modality: str) -> str:
    """Generates a base name: Date_Modality_Timestamp."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = int(time.time())
    safe_modality = str(modality).replace('/', '-')
    return f"{date_str}_{safe_modality}_{timestamp}"


def parse_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Action Recognition Pipeline")
    parser.add_argument('--hypes', default=DEFAULT_CONFIG_PATH, type=str, help='Path to the configuration JSON file.')
    parser.add_argument('--phase', default='train', type=str, choices=['train', 'test'],
                        help='Execution phase: train or test.')
    parser.add_argument('--gpus', default='0', type=str,
                        help='Physical GPU IDs to use, separated by commas (e.g., "0,1").')
    parser.add_argument('--resume', default="", type=str, help='Path to the checkpoint for resuming or testing.')
    parser.add_argument('--note', default='', type=str, help='Additional experimental notes.')

    # Testing specific arguments
    parser.add_argument('--benchmark', action='store_true',
                        help='[Test Only] Run efficiency benchmark (Latency/FPS/Throughput).')
    parser.add_argument('--accuracy', action='store_true',
                        help='[Test Only] Run accuracy evaluation on the validation set.')
    parser.add_argument('--extract', action='store_true',
                        help='[Test Only] Extract targeted features for qualitative analysis.')
    parser.add_argument('--matrix', action='store_true', help='[Test Only] Generate and save the confusion matrix.')

    # Automatically passed by torchrun for DDP
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get("LOCAL_RANK", -1)))

    return parser.parse_args()


def init_distributed_env(args: argparse.Namespace) -> argparse.Namespace:
    """Initializes the Distributed Data Parallel (DDP) environment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.distributed = True
    else:
        # Single GPU / CPU fallback mode
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()

    return args


def set_device(args: argparse.Namespace) -> argparse.Namespace:
    """Assigns the appropriate computation device."""
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.local_rank}')
    else:
        args.device = torch.device('cpu')
    return args


def init_config_and_seed(args: argparse.Namespace) -> Configer:
    """Initializes the configuration manager and sets global seeds."""
    configer = Configer(args)
    seed = configer.get('seed', default=DEFAULT_SEED)
    if args.rank == 0:
        print(f"Global Seed set to: {seed}")
    set_seed(seed)
    return configer


def generate_task_name(args: argparse.Namespace, configer: Configer) -> str:
    """Generates and broadcasts a unique task name across all processes."""
    task_name = "default"
    if args.rank == 0:
        modality = configer.get("data", "type", default=DEFAULT_MODALITY)
        task_name = _generate_timestamp_name(modality)

    if args.distributed:
        object_list = [task_name]
        dist.broadcast_object_list(object_list, src=0)
        task_name = object_list[0]

    return task_name


def build_work_dir(configer: Configer, task_name: str) -> str:
    """Constructs the output directory structure and injects it into configer."""
    output_root = configer.get('checkpoints', 'save_dir', default=DEFAULT_OUTPUT_DIR)
    dataset_name = configer.get('dataset', default=DEFAULT_DATASET)
    modality = configer.get('data', 'type', default=DEFAULT_MODALITY)

    group_dir = f"{dataset_name}_{modality}"
    work_dir = os.path.join(output_root, group_dir, task_name)

    # Inject paths back into configer for downstream modules
    configer.set(task_name, "task_name")
    configer.set(work_dir, "work_dir")

    return work_dir


def backup_code(work_dir: str, rank: int) -> None:
    """Creates a backup of the source code to ensure exact reproducibility."""
    if rank != 0:
        return

    os.makedirs(work_dir, exist_ok=True)
    try:
        backup_dir = os.path.join(work_dir, 'code_backup')
        os.makedirs(backup_dir, exist_ok=True)

        for item_path in BACKUP_SOURCE_DIRS:
            if not os.path.exists(item_path):
                continue

            dest_path = os.path.join(backup_dir, os.path.basename(item_path))
            if os.path.isdir(item_path):
                if os.path.basename(item_path) == '__pycache__':
                    continue
                shutil.copytree(
                    item_path,
                    dest_path,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git*')
                )
            else:
                shutil.copy2(item_path, dest_path)

    except Exception as e:
        print(f"Error during code backup: {e}")


def init_logger(work_dir: str, args: argparse.Namespace, configer: Configer) -> logging.Logger:
    """Initializes the logging system."""
    project_name = configer.get('name', default=DEFAULT_PROJECT_NAME)
    logger = setup_logger(
        output_dir=work_dir,
        distributed_rank=args.rank,
        filename=f'{os.path.basename(work_dir)}.txt',
        name=project_name
    )

    if args.rank == 0:
        logger.info(f"Project: {project_name}")
        logger.info(f"Work Directory: {work_dir}")
        logger.info(f"Physical GPUs: {args.gpus}")
        logger.info(f"Logical World Size: {args.world_size}")
        logger.info(f"Configuration:\n{configer}")

    return logger


def run_task(args: argparse.Namespace, configer: Configer, logger: logging.Logger) -> None:
    """Executes the core training or testing logic."""
    try:
        if args.phase == 'train':
            trainer = GestureTrainer(configer, logger)
            trainer.init_model()
            trainer.train()

        elif args.phase == 'test':
            tester = GestureTester(configer, logger)
            tester.init_model()

            if not (args.benchmark or args.accuracy or args.extract):
                logger.error(
                    "Error: Please specify at least one test task: '--benchmark', '--accuracy', or '--extract'.")
                raise ValueError("No test task specified.")

            if args.benchmark:
                if args.rank == 0:
                    logger.info(">>> [Task] Running Efficiency Benchmark <<<")
                tester.run_efficiency_benchmark()

            if args.accuracy or args.extract:
                tester.init_dataloader()

            if args.accuracy:
                if args.rank == 0:
                    logger.info(">>> [Task] Running Accuracy Evaluation <<<")
                tester.run_accuracy_test(save_confusion_matrix=args.matrix)

            if args.extract:
                if args.rank == 0:
                    logger.info(">>> [Task] Extracting Target Sample Features <<<")
                tester.extract_target_sample_features(target_class_id=REBUTTAL_TARGET_ACTION_ID)

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise


def cleanup_distributed(args: argparse.Namespace) -> None:
    """Safely destroys the distributed process group."""
    if args.distributed:
        dist.destroy_process_group()


# =====================================================================
# Main Execution Flow
# =====================================================================
def main() -> None:
    # 1. Parse Arguments
    args = parse_args()
    # 2. Initialize Distributed Environment
    args = init_distributed_env(args)
    # 3. Assign Devices
    args = set_device(args)
    # 4. Initialize Config and Random Seeds
    configer = init_config_and_seed(args)
    # 5. Generate Task Name
    task_name = generate_task_name(args, configer)
    # 6. Build Output Directories
    work_dir = build_work_dir(configer, task_name)
    # 7. Backup Source Code (Rank 0 only)
    backup_code(work_dir, args.rank)

    # 8. Synchronize Processes
    if args.distributed:
        dist.barrier()

    # 9. Initialize Logger
    logger = init_logger(work_dir, args, configer)

    # 10. Execute Core Task
    try:
        run_task(args, configer, logger)
    finally:
        # 11. Resource Cleanup
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
