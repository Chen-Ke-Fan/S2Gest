import logging
import os
import sys
from typing import Optional

# Extracted constants to avoid magic strings inside the function
DEFAULT_LOG_FORMAT = '[%(asctime)s] %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logger(
    output_dir: Optional[str] = None,
    distributed_rank: int = 0,
    filename: str = "train.log",
    name: str = "Mamba_DHGR"
) -> logging.Logger:
    """
    Initializes and configures the Logger.

    Args:
        output_dir: Path to the directory where the log file will be saved.
                    If None, logs are only printed to the console.
        distributed_rank: Rank ID for distributed training. Only Rank 0 writes to files and console.
        filename: Name of the log file.
        name: Name of the logger instance.

    Returns:
        The configured logging.Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to prevent duplicate printing in Notebooks or multiple calls
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define the logging format
    formatter = logging.Formatter(
        DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT
    )

    # Only the main process (Rank 0) or single-GPU mode performs actual logging.
    # Processes with Rank > 0 remain silent to avoid log clutter.
    if distributed_rank == 0:
        # 1. Console Handler (Stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler (File)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)

            # mode='a' for append, 'w' for overwrite
            file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        # Add NullHandler for non-Rank 0 processes to prevent "No handlers could be found" warnings
        logger.addHandler(logging.NullHandler())

    return logger