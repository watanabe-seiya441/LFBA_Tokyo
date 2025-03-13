import os
import time
import threading
import queue
import logging
import shutil
import random

# Configure logging
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

def create_dataset(watch_dir: str, dataset_dir: str, threshold: int) -> None:
    """
    Processes images from watch_dir and organizes them into a dataset.

    Moves only a portion of images at a time to avoid infinite loops. 
    Splits data into training and validation sets with an 80/20 ratio.

    Args:
        watch_dir (str): Directory to monitor for new images.
        dataset_dir (str): Directory to store the processed dataset.
        threshold (int): Number of images to process in one batch.
    """
    logger.info("[INFO] Dataset creation started.")
    random.seed(57)

    label_dict = {}
    train_folder = os.path.join(dataset_dir, "train")
    val_folder = os.path.join(dataset_dir, "val")

    all_files = [f for f in os.listdir(watch_dir) if f.endswith(".jpg")]
    num_files_to_process = min(len(all_files), threshold)

    logger.info(f"[INFO] Processing {num_files_to_process} images.")

    for filename in all_files[:num_files_to_process]:
        parts = filename.split("_")
        if len(parts) < 2:
            continue

        label = parts[1].split(".")[0]
        label_dict.setdefault(label, []).append(os.path.join(watch_dir, filename))

    for label, files in label_dict.items():
        random.shuffle(files)
        split_index = int(len(files) * 0.8)
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

def monitor_folder(stop_event: threading.Event, start_train: queue.Queue, watch_dir: str, dataset_dir: str, threshold: int, check_interval: int) -> None:
    """
    Monitors the specified folder and creates a dataset if the file count exceeds the threshold.

    Args:
        stop_event (threading.Event): Event to signal the stop of monitoring.
        start_train (queue.Queue): Queue to signal when to start training.
        watch_dir (str): Directory to monitor for new files.
        dataset_dir (str): Directory where the dataset will be saved.
        threshold (int): File count threshold to trigger dataset creation.
        check_interval (int): Time interval in seconds to check the folder.
    """
    last_check_time = 0

    while not stop_event.is_set():
        current_time = time.time()

        if current_time - last_check_time >= check_interval:
            file_count = count_files(watch_dir)
            logger.info(f"[INFO] Current file count: {file_count}")

            if file_count >= threshold:
                logger.warning(f"[WARNING] ⚠️ File count exceeded threshold ({threshold}): {file_count} files")
                create_dataset(watch_dir, dataset_dir, threshold)
                start_train.put("start")

            last_check_time = current_time
