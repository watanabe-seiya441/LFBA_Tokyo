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

from tool.camera.camera import Camera
from tool.serial.communication import SerialCommunication
from tool.serial.serial_reader import listen_serial
from tool.serial.serial_write import write_serial
from tool.camera.cameraApp import capture_latest_frame, save_images
from tool.image_classification.classifier import process_images, ImageClassifier, _classify_image
from tool.create_dataset.folder_monitor import monitor_folder
from tool.model_training.model_training import train_controller

print("[INFO] All packages loaded successfully.")

# Load configuration from config.toml
with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)

# Configuration parameters
SERIAL_PORT = config["serial"]["port"] # "/dev/ttyACM0"
BAUDRATE = config["serial"]["baudrate"] # 9600
cameraID = config["camera"]["cameraID"] # 0
CLASSES = config["model"]["classes"] # ["0000", "0001", "0011", "0100", "0110", "1000", "1010", "1111"]
MODEL_NAME = config["model"]["name"] # "dataset0b_vgg_best_model.pth"
arch = config["model"]["arch"] # "vgg16"
is_update = config["model"]["is_update"] # false
BATCH_SIZE = config["hyperparameters"]["batch_size"] # 32
EPOCHS = config["hyperparameters"]["epochs"] # 5
IMG_SIZE = config["hyperparameters"]["img_size"] # 224
LEARNING_RATE = config["hyperparameters"]["learning_rate"] # 0.001
DATASET_DIR = config["directory"]["dataset_dir"] # "dataset"
MODEL_DIR = config["directory"]["model_dir"] # "model"
GPU = config["gpu"]["gpu_index"] # 1
IMAGE_DIR = config["directory"]["image_dir"] # "image/test"
THRESHOLD = config["monitoring"]["THRESHOLD"] # 10000 # flie count threshold
CHECK_INTERVAL = config["monitoring"]["CHECK_INTERVAL"] # 60 

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)  # e.g. "model/best_model.pth"

# Thread management events
stop_event = threading.Event()  # e.g. Event() - system shutdown signal
mode_train = threading.Event()  # e.g. Event() - training mode flag
mode_record = threading.Event()  # e.g. Event() - frame recording mode flag
# mode_train.set()  # 推論を有効にするためにコメントアウト
mode_record.set()  # フレームキャプチャを有効にする

# Queue definitions
read_queue = queue.Queue()  # e.g. Queue() - serial read data
write_queue = queue.Queue()  # e.g. Queue() - serial write data
frame_queue = queue.Queue(maxsize=1)  # e.g. Queue() - camera frames for inference
api_frame_queue = queue.Queue(maxsize=1)  # e.g. Queue() - API専用のフレームキュー
image_queue = queue.Queue()  # e.g. Queue() - images for dataset creation
label_queue = queue.Queue(maxsize=1)  # e.g. Queue() - classification labels
classes_queue = queue.Queue()  # e.g. Queue() - class names list
start_train = queue.Queue()  # e.g. Queue() - training start signal
start_train.put("start")  # e.g. "start" - initial training trigger

# 最新の推論結果を保存するグローバル変数
latest_inference_bits = "----"  # e.g. "0010", "1101" - 4-bit classification result

# Logging setup
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
start_time = datetime.now().strftime("%Y%m%dT%H%M%S")  # e.g. "20241215T143052"
log_filename = f"{log_dir}/system_{start_time}.log"  # e.g. "log/system_20241215T143052.log"

# ファイルログの設定
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# コンソールログの設定（INFO以上のみ）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)

def handle_received_data():
    """
    Process received serial data and update label queue.
    """
    while not stop_event.is_set():
        try:
            received_data = read_queue.get(timeout=0.1)  # e.g. "A12340" - serial data string
            mode_train.set() if received_data[5] == "0" else mode_train.clear()  # e.g. received_data[5] = "0" or "1"
            mode_record.set() if received_data[6] == "1" else mode_record.clear()  # e.g. received_data[6] = "0" or "1"
            latest_label = received_data[1:5]  # e.g. "1234" - 4-digit label
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
        user_input = input("> ").strip()  # e.g."quit", "test command"
        if user_input.lower() in ["q", "quit"]:
            print("[INFO] Stopping the system...")
            stop_event.set()
            break
        write_queue.put(user_input)

def distribute_frames(camera, frame_queue, api_frame_queue, stop_event, mode_record):
    """
    カメラからのフレームを推論用とAPI用の両方のキューに配信
    """
    logger.info("[FRAME] Frame distribution started")
    frame_count = 0  # number of processed frames
    while not stop_event.is_set():
        try:
            # mode_recordの状態を確認（元のcapture_latest_frameと同様）
            if not mode_record.is_set():
                logger.warning("[FRAME] mode_record is not set, skipping frame capture")
                time.sleep(0.1)
                continue
                
            frame = camera.capture_frame()  # e.g. numpy.ndarray(480, 640, 3) - BGR image
            if frame is not None:
                frame_count += 1
                
                # 推論用キューに送信
                if not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                frame_queue.put_nowait(frame)
                
                # API用キューに送信
                if not api_frame_queue.empty():
                    try:
                        api_frame_queue.get_nowait()
                    except:
                        pass
                api_frame_queue.put_nowait(frame)
                
                # 10フレームごとにログ出力
                if frame_count % 10 == 0:
                    logger.debug(f"[FRAME] {frame_count} frames processed")
            else:
                logger.warning("[FRAME] Failed to capture frame")
                
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"[FRAME] Distribution error: {e}")
            time.sleep(0.1)

def start_threads(serial_comm, camera):
    threads = [
        threading.Thread(target=listen_serial, args=(stop_event, serial_comm, read_queue), daemon=True),
        threading.Thread(target=write_serial, args=(stop_event, serial_comm, write_queue), daemon=True),
        threading.Thread(target=distribute_frames, args=(camera, frame_queue, api_frame_queue, stop_event, mode_record), daemon=True),
        threading.Thread(target=save_images, args=(stop_event, mode_train, frame_queue, image_queue, camera, label_queue, start_time, IMAGE_DIR), daemon=True),
        threading.Thread(target=process_images_with_bits, args=(stop_event, mode_train, frame_queue, write_queue, MODEL_PATH, CLASSES, classes_queue), daemon=True),
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

    for thread in threads:
        thread.start()
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

# frameとlatest_inference_bitsをフロントに返す
# note: process_imagesの中でやるべきかも
@app.get("/api/inference")
def api_inference():
    logger.debug(f"[API] Request received, bits: {latest_inference_bits}")
    
    try:
        frame = api_frame_queue.get_nowait()  # e.g. numpy.ndarray(480, 640, 3) - BGR image
        logger.debug("[API] Frame retrieved from queue")
    except queue.Empty:
        logger.debug("[API] No frame available")
        return {"imageUrl": "", "bits": latest_inference_bits}
        
    import cv2, base64
    ok, buffer = cv2.imencode('.jpg', frame)  # e.g. ok=True, buffer=encoded_bytes
    if not ok:
        logger.warning("[API] Failed to encode frame")
        return {"imageUrl": "", "bits": latest_inference_bits}
        
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
    logger.debug(f"[API] Response sent, bits: {latest_inference_bits}")
    return {"imageUrl": data_uri, "bits": latest_inference_bits}

# note: process_imagesの中でやるべきかも
def process_images_with_bits(stop_event, mode_train, frame_queue, write_queue, model_path, classes, classes_queue):
    """
    Process images and update global inference bits variable.
    """
    global latest_inference_bits
    
    classifier, previous_prediction, consecutive_count = None, None, 0  # e.g. None, 2, 3
    last_model_update = None  # e.g. None, 1640995200.123 - timestamp
    current_state = None  # e.g. None, "0000", "0001", ...
    inference_count = 0
    
    logger.info("[INFERENCE] Image processing started")

    while not stop_event.is_set():
        # モデルの更新チェック（元のコードから流用）
        if os.path.exists(model_path):
            current_update = os.path.getmtime(model_path)  # e.g. 1640995200.123 - file modification timestamp
            if not classifier or current_update > (last_model_update or 0):
                logger.info(f"[INFERENCE] Model loading from {model_path}")
                try:
                    classifier = ImageClassifier(model_path)  # e.g. ImageClassifier instance
                    logger.info("[INFERENCE] Classifier loaded successfully")
                except Exception as e:
                    logger.error(f"[INFERENCE] Failed to load classifier: {e}")
                    classifier = None
                
                if not classes_queue.empty():
                    classes = classes_queue.get()  # e.g. ["0000", "0001", ...]
                    classes_queue.put(classes)
                    logger.info(f"[INFERENCE] Classes loaded: {len(classes)} classes")
                    
                last_model_update = current_update
        else:
            logger.warning(f"[INFERENCE] Model path does not exist: {model_path}")

        if not classifier:
            logger.warning("[INFERENCE] No classifier available, waiting...")
            time.sleep(1)
            continue

        # 強制的に推論を実行（mode_trainの状態を無視）
        logger.debug("[INFERENCE] Processing images...")

        # フレームキューから画像を取得（タイムアウト付き）
        try:
            image_data = frame_queue.get(timeout=0.1)
            inference_count += 1

            if image_data is None:
                logger.debug("[INFERENCE] Image data is None")
                frame_queue.task_done()
                continue

            predicted_class, confidence = _classify_image(classifier, image_data)
            
            if predicted_class is None or confidence is None:
                logger.debug("[INFERENCE] Classification returned None")
                frame_queue.task_done()
                continue

            # 推論結果を4bit形式で保存
            if len(classes) <= 16:  # 4bitで表現可能な範囲
                bits_result = format(predicted_class, '04b')
                latest_inference_bits = bits_result
                
                # 5回に1回だけログ出力
                if inference_count % 5 == 0:
                    logger.debug(f"[INFERENCE] Class: {predicted_class}, Confidence: {confidence:.2f}, Bits: {bits_result}")
            else:
                logger.warning(f"[INFERENCE] Too many classes ({len(classes)}) for 4-bit representation")

            consecutive_count = consecutive_count + 1 if predicted_class == previous_prediction else 1 
            if consecutive_count >= 3 and current_state != classes[predicted_class]:
                write_queue.put(classes[predicted_class])  # e.g. "0000", "0001", ...
                logger.info(f"[INFERENCE] Stable prediction: {classes[predicted_class]} (Confidence: {confidence:.2f})")
                current_state = classes[predicted_class]  # e.g. "0000", "0001", ...
                time.sleep(5)
            previous_prediction = predicted_class
            frame_queue.task_done()
            
        except queue.Empty:
            logger.debug("[INFERENCE] Frame queue empty, waiting...")
            time.sleep(0.1)
            continue
        except Exception as e:
            logger.error(f"[INFERENCE] Exception: {e}")
            time.sleep(0.1)
            continue
            
        time.sleep(0.1)  # CPU使用率を下げるため

def main():
    serial_comm = SerialCommunication(SERIAL_PORT, BAUDRATE)  # e.g. SerialCommunication("/dev/ttyUSB0", 9600)
    camera = Camera(camera_id=cameraID, capture_interval=1)  # e.g. Camera(camera_id=0, capture_interval=1)

    threads = start_threads(serial_comm, camera)

    # <追記> FastAPI サーバーを別スレッドで起動
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info"),
        daemon=True
    )
    api_thread.start()

    threads[-1].join()  # e.g. wait for user_input_listener thread
    stop_event.set()
    for thread in threads[:-1]:
        thread.join()

    serial_comm.close()
    camera.release()
    print("[INFO] System shutdown.")

if __name__ == "__main__":
    main()
