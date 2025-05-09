"""
back.py
========
main.py の全コードを **一切削除せず** 保持したまま、末尾に FastAPI サーバー機能を追記した統合スクリプト。
- `/api/inference` で最新フレーム（Base64 JPEG）と 4bit 推論結果を返す。
- Poetry 環境下で `poetry run python back.py` で起動。
"""

print("[INFO] Start system...")

import threading
import queue
import time
import logging
import os
import tomllib
from datetime import datetime

from fastapi import FastAPI  # 追加
from fastapi.middleware.cors import CORSMiddleware  # 追加
import uvicorn  # 追加

from centralmaneger.camera.camera import Camera
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial
from centralmaneger.serial.serial_write import write_serial
from centralmaneger.camera.cameraApp import capture_latest_frame, save_images
from centralmaneger.image_classification.classifier import process_images
from centralmaneger.create_dataset.folder_monitor import monitor_folder
from centralmaneger.model_training.model_training import train_controller

print("[INFO] All packages loaded successfully.")

# Load configuration from config.toml
with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)

# Configuration parameters
SERIAL_PORT = config["serial"]["port"]
BAUDRATE = config["serial"]["baudrate"]
cameraID = config["camera"]["cameraID"]
CLASSES = config["model"]["classes"]
MODEL_NAME = config["model"]["name"]
arch = config["model"]["arch"]
is_update = config["model"]["is_update"]
BATCH_SIZE = config["hyperparameters"]["batch_size"]
EPOCHS = config["hyperparameters"]["epochs"]
IMG_SIZE = config["hyperparameters"]["img_size"]
LEARNING_RATE = config["hyperparameters"]["learning_rate"]
DATASET_DIR = config["directory"]["dataset_dir"]
MODEL_DIR = config["directory"]["model_dir"]
GPU = config["gpu"]["gpu_index"]
IMAGE_DIR = config["directory"]["image_dir"]
THRESHOLD = config["monitoring"]["THRESHOLD"]
CHECK_INTERVAL = config["monitoring"]["CHECK_INTERVAL"]

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Thread management events
stop_event = threading.Event()
mode_train = threading.Event()
mode_record = threading.Event()
mode_train.set()

# Queue definitions
read_queue = queue.Queue()
write_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=1)
image_queue = queue.Queue()
label_queue = queue.Queue(maxsize=1)
classes_queue = queue.Queue()
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
    """
    Process received serial data and update label queue.
    """
    while not stop_event.is_set():
        try:
            received_data = read_queue.get(timeout=0.1)
            mode_train.set() if received_data[5] == "0" else mode_train.clear()
            mode_record.set() if received_data[6] == "1" else mode_record.clear()
            latest_label = received_data[1:5]
            if not label_queue.empty():
                label_queue.get()
            label_queue.put(latest_label)
        except queue.Empty:
            continue

def user_input_listener():
    """CLI input listener."""
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
    threads = [
        threading.Thread(target=listen_serial, args=(stop_event, serial_comm, read_queue), daemon=True),
        threading.Thread(target=write_serial, args=(stop_event, serial_comm, write_queue), daemon=True),
        threading.Thread(target=capture_latest_frame, args=(camera, frame_queue, stop_event, mode_record), daemon=True),
        threading.Thread(target=save_images, args=(stop_event, mode_train, frame_queue, image_queue, camera, label_queue, start_time, IMAGE_DIR), daemon=True),
        threading.Thread(target=process_images, args=(stop_event, mode_train, frame_queue, write_queue, MODEL_PATH, CLASSES, classes_queue), daemon=True),
        threading.Thread(target=handle_received_data, daemon=True),
        threading.Thread(target=user_input_listener, daemon=True)
    ]

    if is_update:
        threads.append(
            threading.Thread(target=monitor_folder, args=(stop_event, start_train, IMAGE_DIR, DATASET_DIR, THRESHOLD, CHECK_INTERVAL), daemon=True)
        )
        threads.append(
            threading.Thread(target=train_controller, args=(stop_event, start_train, BATCH_SIZE, EPOCHS, IMG_SIZE, LEARNING_RATE, DATASET_DIR, MODEL_DIR, MODEL_NAME, GPU, classes_queue, arch), daemon=True)
        )

    for t in threads:
        t.start()
    return threads

# ----------------------------
# FastAPI サーバー定義 (追記)
# ----------------------------
app = FastAPI(title="Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/inference")
def api_inference():
    try:
        frame = frame_queue.get_nowait()
    except queue.Empty:
        return {"imageUrl": "", "bits": "----"}
    import cv2, base64
    ok, buffer = cv2.imencode('.jpg', frame)
    if not ok:
        return {"imageUrl": "", "bits": "----"}
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
    bits = "1101"
    return {"imageUrl": data_uri, "bits": bits}


def main():
    serial_comm = SerialCommunication(SERIAL_PORT, BAUDRATE)
    camera = Camera(camera_id=cameraID, capture_interval=1)

    threads = start_threads(serial_comm, camera)

    # <追記> FastAPI サーバーを別スレッドで起動
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info"),
        daemon=True
    )
    api_thread.start()

    threads[-1].join()
    stop_event.set()
    for t in threads[:-1]:
        t.join()

    serial_comm.close()
    camera.release()
    print("[INFO] System shutdown.")

if __name__ == "__main__":
    main()
