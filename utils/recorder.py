import os
import csv
import datetime
from typing import Any

# Extracted constants to avoid magic strings and hardcoded configurations
CSV_HEADERS = (
    "Timestamp",
    "Task Name",
    "Dataset",
    "Modality",
    "Note",
    "Best Accuracy",
    "Status"
)
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def record_experiment(
        file_path: str,
        task_name: str,
        note: str,
        dataset: str,
        modality: str,
        best_acc: float,
        status: str = "Finished"
) -> None:
    """
    Records the experiment configuration and results to a CSV file.

    Args:
        file_path: The absolute or relative path to the target CSV file.
        task_name: The name of the current experiment task.
        note: Additional notes or remarks regarding this run.
        dataset: The name of the dataset used.
        modality: The data modality (e.g., RGB, Depth, Skeleton).
        best_acc: The best accuracy achieved during the training/evaluation.
        status: The final status of the experiment (default: "Finished").
    """
    # Prepare the data dictionary to be written
    timestamp = datetime.datetime.now().strftime(DATETIME_FORMAT)
    row_data = {
        "Timestamp": timestamp,
        "Task Name": task_name,
        "Dataset": dataset,
        "Modality": modality,
        "Note": note,
        "Best Accuracy": f"{best_acc:.4f}",
        "Status": status
    }

    # Check if the file already exists to determine if we need to write headers
    file_exists = os.path.isfile(file_path)

    try:
        # Open the file in append mode ('a') with 'utf-8-sig' for Excel compatibility
        with open(file_path, mode='a', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)

            # Write the header row only if it's a newly created file
            if not file_exists:
                writer.writeheader()

            writer.writerow(row_data)
            print(f"✅ Experiment log successfully saved to: {file_path}")

    except Exception as e:
        print(f"❌ Failed to write experiment log: {e}")