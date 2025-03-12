import os
import time
import threading
import queue
import logging
import shutil
import random

# Logging setup
logger = logging.getLogger(__name__)

def count_files(directory: str) -> int:
    """
    Counts the number of files in the specified directory.

    Args:
        directory (str): The directory path to count files in.

    Returns:
        int: The number of files in the directory.
    """
    try:
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    except Exception as e:
        logger.error(f"[ERROR] Failed to read directory: {e}")
        return 0

def create_dataset(WATCH_DIR: str, dataset_dir: str):
    """
    Processes images from WATCH_DIR and organizes them into a dataset.
    Moves only a portion of images at a time to avoid infinite loops.
    """
    logger.info("[INFO] Dataset creation started.")

    # Fix random seed for reproducibility
    random.seed(57)

    label_dict = {}
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")

    # Process only a limited number of images from WATCH_DIR
    all_files = [f for f in os.listdir(WATCH_DIR) if f.endswith(".jpg")]
    num_files_to_process = min(len(all_files), THRESHOLD)  # 半分だけ処理

    logger.info(f"[INFO] Processing {num_files_to_process} images.")

    # Collect label information
    for filename in all_files[:num_files_to_process]:  # Only process a subset
        parts = filename.split("_")
        if len(parts) < 2:
            continue  # Ignore incorrectly formatted files

        label = parts[1].split(".")[0]  # Extract label

        if label not in label_dict:
            label_dict[label] = []

        label_dict[label].append(os.path.join(WATCH_DIR, filename))

    # Distribute images into train and val folders
    for label, files in label_dict.items():
        random.shuffle(files)
        split_index = int(len(files) * 0.8)  # 80% train, 20% val

        train_files = files[:split_index]
        val_files = files[split_index:]

        train_label_folder = os.path.join(train_folder, label)
        val_label_folder = os.path.join(val_folder, label)

        os.makedirs(train_label_folder, exist_ok=True)
        os.makedirs(val_label_folder, exist_ok=True)

        for file in train_files:
            shutil.move(file, os.path.join(train_label_folder, os.path.basename(file)))

        for file in val_files:
            shutil.move(file, os.path.join(val_label_folder, os.path.basename(file)))

    logger.info("[INFO] Dataset creation completed successfully.")

def monitor_folder(stop_event: threading.Event, start_train: queue.Queue, WATCH_DIR: str, dataset_dir: str, THRESHOLD: int, CHECK_INTERVAL: int) -> None:
    """
    Monitors the specified folder and creates a dataset if the file count exceeds the threshold.
    This function runs in a continuous loop, but checks file count only every CHECK_INTERVAL seconds.
    """
    last_check_time = 0  # Store the last file count check time

    while not stop_event.is_set():
        current_time = time.time()

        # Check file count only if CHECK_INTERVAL has passed
        if current_time - last_check_time >= CHECK_INTERVAL:
            file_count = count_files(WATCH_DIR)
            logger.info(f"[INFO] Current file count: {file_count}")

            if file_count >= THRESHOLD:
                logger.warning(f"[WARNING] ⚠️ File count exceeded threshold ({THRESHOLD}): {file_count} files")
                create_dataset(dataset_dir)  # Create dataset immediately
                start_train.put("start")

            last_check_time = current_time  # Update last check time

        # Do not sleep, allowing while-loop to be responsive
