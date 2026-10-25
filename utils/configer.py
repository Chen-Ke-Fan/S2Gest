import json
import os
import argparse
import re
from typing import Any, Dict, Union, Tuple


class Configer:
    """
    Configuration Manager.

    Responsible for loading, merging, and retrieving configuration parameters.
    Priority strategy: Command-line arguments (Args) > Configuration file parameters (JSON File).
    """

    def __init__(self, args: Union[argparse.Namespace, Dict[str, Any]]) -> None:
        """
        Initialize the configuration manager.

        Args:
            args: argparse.Namespace object or dictionary containing command-line arguments.
        """
        if isinstance(args, argparse.Namespace):
            self.args = vars(args)
        else:
            self.args = args

        self.params: Dict[str, Any] = {}

        # Load JSON configuration file
        hypes_path = self.args.get('hypes')
        if hypes_path:
            if not os.path.exists(hypes_path):
                raise FileNotFoundError(f"Configuration file path does not exist: {hypes_path}")

            with open(hypes_path, 'r', encoding='utf-8') as f:
                self.params = json.load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieve a configuration item, supporting deep nested queries.

        Args:
            *keys: Sequence of keys, e.g., configer.get('data', 'train', 'batch_size')
            default: Default value returned if the key sequence is not found.

        Returns:
            The retrieved value or the default value.
        """
        if len(keys) == 0:
            return self.params

        # 1. Prioritize checking command-line arguments
        last_key = keys[-1]
        if last_key in self.args and self.args[last_key] is not None:
            return self.args[last_key]

        # 2. Check the configuration file (recursive search)
        current_node = self.params
        for key in keys:
            if isinstance(current_node, dict) and key in current_node:
                current_node = current_node[key]
            else:
                return default

        return current_node

    def set(self, value: Any, *keys: str) -> None:
        """
        Dynamically modify or add a configuration item.
        Typically used to inject paths, task names, etc., during runtime.

        Args:
            value: The value to set.
            *keys: Sequence of keys, e.g., configer.set(path, 'checkpoints', 'save_dir')
        """
        if not keys:
            raise ValueError("Configer.set requires at least one key.")

        # If modifying a top-level key that exists in args, prioritize modifying args
        last_key = keys[-1]
        if len(keys) == 1 and last_key in self.args:
            self.args[last_key] = value
            return

        # Otherwise, modify params (recursively create dictionaries if necessary)
        current_node = self.params
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                current_node[key] = value
            else:
                if key not in current_node:
                    current_node[key] = {}
                current_node = current_node[key]

                if not isinstance(current_node, dict):
                    raise TypeError(f"Config path conflict at key '{key}'. Expected dict, got {type(current_node)}.")

    def __getitem__(self, item: Union[str, Tuple[str, ...]]) -> Any:
        if isinstance(item, tuple):
            return self.get(*item)
        else:
            return self.get(item)

    def __getattr__(self, item: str) -> Any:
        """Allow accessing top-level attributes via dot notation."""
        return self.get(item)

    @staticmethod
    def _format_compact_json(data: Dict[str, Any]) -> str:
        """
        Helper function to format JSON compactly for clean terminal logging.
        """
        text = json.dumps(data, indent=4, ensure_ascii=False, default=str)

        def collapse(match: re.Match) -> str:
            content = match.group(1)
            content = re.sub(r'\s+', ' ', content).strip()
            return f"[{content}]"

        text = re.sub(r'\[\s*([^\[\]\{\}]+?)\s*\]', collapse, text)
        return text

    def __str__(self) -> str:
        """
        String representation of the configurations for logging/printing.
        """
        info = "\n================ Configuration ================\n"
        info += "--- Arguments (Command Line) ---\n"

        # Filter out private attributes
        clean_args = {k: v for k, v in self.args.items() if not k.startswith('_')}
        info += Configer._format_compact_json(clean_args)

        info += "\n\n--- Parameters (JSON File) ---\n"
        info += Configer._format_compact_json(self.params)
        info += "\n===============================================\n"
        return info