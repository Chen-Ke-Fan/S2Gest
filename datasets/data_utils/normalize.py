import numpy as np

# Centralized constant for numerical stability during division
NUMERICAL_EPSILON: float = 1e-6

def normalize(
    tensor: np.ndarray,
    channel_axis: int = 2,
    epsilon: float = NUMERICAL_EPSILON
) -> np.ndarray:
    """
    Applies channel-wise Z-score normalization to a given n-dimensional tensor.

    This function dynamically computes the mean and standard deviation across all
    dimensions except the specified channel axis. It provides defensive mechanisms
    against zero-division and arbitrary tensor shapes, ensuring robust execution.

    Args:
        tensor (np.ndarray): The input numerical array to be normalized.
        channel_axis (int): The index of the channel dimension. Defaults to 2.
        epsilon (float): A small constant used to prevent division by zero.

    Returns:
        np.ndarray: A new tensor containing the normalized values.

    Raises:
        TypeError: If the input is not a valid numpy.ndarray.
        ValueError: If the tensor dimensions cannot accommodate the channel_axis.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected input of type numpy.ndarray, received {type(tensor).__name__}.")

    normalized_tensor = tensor

    if normalized_tensor.ndim < 4:
        try:
            normalized_tensor = np.expand_dims(normalized_tensor, axis=channel_axis)
        except np.AxisError:
            raise ValueError(
                f"Cannot expand tensor of shape {tensor.shape} at axis {channel_axis}."
            )

    # Dynamic Axis Inference: (e.g., for 4D tensor with channel_axis=2, this yields (0, 1, 3))
    reduction_axes = tuple(
        dim for dim in range(normalized_tensor.ndim) if dim != channel_axis
    )

    mean = np.mean(normalized_tensor, axis=reduction_axes, keepdims=True)
    std = np.std(normalized_tensor, axis=reduction_axes, keepdims=True)

    std = np.maximum(std, epsilon)

    return (normalized_tensor - mean) / std