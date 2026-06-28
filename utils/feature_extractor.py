import torch
import torch.nn as nn
from typing import Any, Dict, Optional
import numpy as np


class FeatureExtractor:
    """
    Non-invasive dynamic feature extractor based on context managers.
    Safely attaches and detaches PyTorch forward hooks to extract intermediate tensors
    without modifying the underlying model source code.
    """

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self.model = model
        self.layer_name = layer_name
        self.features: Optional[np.ndarray] = None
        self._hook_handle = None

    def _hook_fn(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        """Core hook function: intercepts and caches the target layer's output."""
        # Handle cases where a layer might return a tuple (e.g., RNNs)
        if isinstance(output, tuple):
            output = output[0]
        # Detach from computational graph and move to CPU
        self.features = output.detach().cpu().numpy()

    def __enter__(self) -> 'FeatureExtractor':
        """Dynamically registers the hook upon entering the context."""
        modules_dict: Dict[str, nn.Module] = dict(self.model.named_modules())

        if self.layer_name not in modules_dict:
            available_layers = list(modules_dict.keys())
            raise ValueError(
                f"❌ Target layer '{self.layer_name}' not found in the model architecture.\n"
                f"Available layers (first 10): {available_layers[:10]}..."
            )

        target_layer = modules_dict[self.layer_name]
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Ensures the hook is safely removed, preventing memory leaks."""
        if self._hook_handle is not None:
            self._hook_handle.remove()