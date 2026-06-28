from typing import Union


class AverageMeter:
    """
    Computes and stores the average and current value.
    Frequently used for tracking metrics such as loss or accuracy over iterations/epochs.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Resets all internal metric accumulators to zero.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: Union[float, int], n: int = 1) -> None:
        """
        Updates the meter with a new recorded value.

        Args:
            val: The current metric value to record.
            n: The number of instances this value represents (typically the batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count