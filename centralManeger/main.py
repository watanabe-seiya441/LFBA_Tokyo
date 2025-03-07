print("[INFO] Start system...")

import threading
import queue
import time
import logging
import os
import tomllib
from datetime import datetime

from centralmaneger.camera.camera import Camera
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial
from centralmaneger.serial.serial_write import write_serial
from centralmaneger.camera.cameraApp import capture_latest_frame, save_images
from centralmaneger.image_classification.classifier import process_images
from centralmaneger.create_dataset.folder_monitor import monitor_folder  # monitor_folder を追加
from centralmaneger.model_training.model_training import train_controller

print("[INFO] All packages loaded successfully.")

# Load configuration from config.toml
with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)

SERIAL_PORT = config["serial"]["port"]
BAUDRATE = config["serial"]["baudrate"]
CLASSES = config["model"]["classes"]
MODEL_PATH = config["model"]["path"]
WATCH_DIR = "image/recorded"  # Folder to monitor

# Thread management events
stop_event = threading.Event()
mode_train = threading.Event()
mode_train.set()

# Queue definitions
read_queue = queue.Queue()
write_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=1)
image_queue = queue.Queue()
label_queue = queue.Queue(maxsize=1)
start_train = queue.Queue() 
start_train.put("start")

# Logging setup
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
start_time = datetime.now().strftime("%Y%m%dT%H%M%S")
log_filename = f"{log_dir}/system_{start_time}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def handle_received_data():
    """ Thread function to process received serial data and update label queue """
    while not stop_event.is_set():
        try:
            received_data = read_queue.get(timeout=0.1)
            mode_train.set() if received_data[5] == "0" else mode_train.clear()
            
            latest_label = received_data[1:5]
            if not label_queue.empty():
                label_queue.get()
            label_queue.put(latest_label)
        except queue.Empty:
            continue

def user_input_listener():
    """ Thread function to handle user input """
    print("[INFO] Enter commands to interact with serial communication.")
    print("[INFO] Type 'q' or 'quit' to exit.")
    while not stop_event.is_set():
        user_input = input("> ").strip()
        if user_input.lower() in ["q", "quit"]:
            print("[INFO] Stopping the system...")
            stop_event.set()
            break
        write_queue.put(user_input)

def start_threads(serial_comm, camera):
    """ Start all necessary threads """
    threads = [
        threading.Thread(target=listen_serial, args=(stop_event, serial_comm, read_queue), daemon=True),
        threading.Thread(target=write_serial, args=(stop_event, serial_comm, write_queue), daemon=True),
        threading.Thread(target=capture_latest_frame, args=(camera, frame_queue, stop_event), daemon=True),
        threading.Thread(target=save_images, args=(stop_event, mode_train, frame_queue, image_queue, camera, label_queue, start_time), daemon=True),
        threading.Thread(target=process_images, args=(stop_event, mode_train, frame_queue, write_queue, MODEL_PATH, CLASSES), daemon=True),
        threading.Thread(target=handle_received_data, daemon=True),
        threading.Thread(target=monitor_folder, args=(stop_event, start_train), daemon=True),  # Add folder monitor thread
        threading.Thread(target=train_controller, args=(stop_event, start_train), daemon=True), 
        threading.Thread(target=user_input_listener, daemon=True)
    ]
    
    for thread in threads:
        thread.start()
    return threads

def main():
    """ Main function to manage the system """
    serial_comm = SerialCommunication(SERIAL_PORT, BAUDRATE)
    camera = Camera(camera_id=0, capture_interval=1)
    
    threads = start_threads(serial_comm, camera)
    
    # Wait for user input thread to finish
    threads[-1].join()
    stop_event.set()
    
    # Ensure all other threads terminate
    for thread in threads[:-1]:
        thread.join()
    
    serial_comm.close()
    camera.release()
    print("[INFO] System shutdown.")

if __name__ == "__main__":
    main()
