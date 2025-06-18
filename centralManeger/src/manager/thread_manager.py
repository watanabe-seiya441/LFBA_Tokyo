"""
thread_manager.py
================
システムのスレッド管理を行うモジュール
"""

import threading
import queue
import logging
import time
from datetime import datetime

from tool.serial.serial_reader import listen_serial
from tool.serial.serial_write import write_serial
from tool.camera.cameraApp import save_images
from tool.create_dataset.folder_monitor import monitor_folder
from tool.model_training.model_training import train_controller

logger = logging.getLogger(__name__)


class ThreadManager:
    """システムのスレッドを管理するクラス"""
    
    def __init__(self, config):
        self.config = config
        self.threads = []
        
        # Thread management events
        self.stop_event = threading.Event()
        self.mode_train = threading.Event()
        self.mode_record = threading.Event()
        self.mode_record.set()  # フレームキャプチャを有効にする
        
        # Queue definitions
        self.read_queue = queue.Queue()
        self.write_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=1)
        self.api_frame_queue = queue.Queue(maxsize=1) # frontに送るためのキュー
        self.image_queue = queue.Queue()
        self.label_queue = queue.Queue(maxsize=1)
        self.classes_queue = queue.Queue()
        self.start_train = queue.Queue()
        self.start_train.put("start")
        
        # 最新の推論結果を保存するためのグローバル変数
        self.latest_inference_bits = "----"
        
        # 開始時刻を記録
        self.start_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    def handle_received_data(self):
        """シリアルデータを処理してラベルキューを更新"""
        while not self.stop_event.is_set():
            try:
                received_data = self.read_queue.get(timeout=0.1)
                self.mode_train.set() if received_data[5] == "0" else self.mode_train.clear()
                self.mode_record.set() if received_data[6] == "1" else self.mode_record.clear()
                latest_label = received_data[1:5]
                if not self.label_queue.empty():
                    self.label_queue.get()
                self.label_queue.put(latest_label)
            except queue.Empty:
                continue
    
    def user_input_listener(self):
        """CLI入力リスナー"""
        print("[INFO] Enter commands to interact with serial communication.")
        print("[INFO] Type 'q' or 'quit' to exit.")
        while not self.stop_event.is_set():
            user_input = input("> ").strip()
            if user_input.lower() in ["q", "quit"]:
                print("[INFO] Stopping the system...")
                self.stop_event.set()
                break
            self.write_queue.put(user_input)
    
    def distribute_frames(self, camera):
        """カメラからのフレームを推論用とAPI用の両方のキューに配信"""
        logger.info("[FRAME] Frame distribution started")
        frame_count = 0
        while not self.stop_event.is_set():
            try:
                if not self.mode_record.is_set():
                    logger.warning("[FRAME] mode_record is not set, skipping frame capture")
                    time.sleep(0.1)
                    continue
                    
                frame = camera.capture_frame()
                if frame is not None:
                    frame_count += 1
                    
                    # 推論用キューに送信
                    if not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put_nowait(frame)
                    
                    # API用キューに送信
                    if not self.api_frame_queue.empty():
                        try:
                            self.api_frame_queue.get_nowait()
                        except:
                            pass
                    self.api_frame_queue.put_nowait(frame)
                    
                    # 10フレームごとにログ出力
                    if frame_count % 10 == 0:
                        logger.debug(f"[FRAME] {frame_count} frames processed")
                else:
                    logger.warning("[FRAME] Failed to capture frame")
                    
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"[FRAME] Distribution error: {e}")
                time.sleep(0.1)
    
    def process_images_with_bits(self):
        """画像を処理してグローバル推論ビット変数を更新"""
        from tool.image_classification.classifier import ImageClassifier, _classify_image
        
        classifier, previous_prediction, consecutive_count = None, None, 0
        last_model_update = None
        current_state = None
        inference_count = 0
        
        logger.info("[INFERENCE] Image processing started")

        while not self.stop_event.is_set():
            # モデルの更新チェック
            import os
            if os.path.exists(self.config.model_path):
                current_update = os.path.getmtime(self.config.model_path)
                if not classifier or current_update > (last_model_update or 0):
                    logger.info(f"[INFERENCE] Model loading from {self.config.model_path}")
                    try:
                        classifier = ImageClassifier(self.config.model_path)
                        logger.info("[INFERENCE] Classifier loaded successfully")
                    except Exception as e:
                        logger.error(f"[INFERENCE] Failed to load classifier: {e}")
                        classifier = None
                    
                    if not self.classes_queue.empty():
                        classes = self.classes_queue.get()
                        self.classes_queue.put(classes)
                        logger.info(f"[INFERENCE] Classes loaded: {len(classes)} classes")
                        
                    last_model_update = current_update
            else:
                logger.warning(f"[INFERENCE] Model path does not exist: {self.config.model_path}")

            if not classifier:
                logger.warning("[INFERENCE] No classifier available, waiting...")
                time.sleep(1)
                continue

            # フレームキューから画像を取得
            try:
                image_data = self.frame_queue.get(timeout=0.1)
                inference_count += 1

                if image_data is None:
                    logger.debug("[INFERENCE] Image data is None")
                    self.frame_queue.task_done()
                    continue

                predicted_class, confidence = _classify_image(classifier, image_data)
                
                if predicted_class is None or confidence is None:
                    logger.debug("[INFERENCE] Classification returned None")
                    self.frame_queue.task_done()
                    continue

                # 推論結果を4bit形式で保存
                classes = self.config.model.classes
                if len(classes) <= 16:
                    bits_result = format(predicted_class, '04b')
                    self.latest_inference_bits = bits_result
                    
                    # 5回に1回だけログ出力
                    if inference_count % 5 == 0:
                        logger.debug(f"[INFERENCE] Class: {predicted_class}, Confidence: {confidence:.2f}, Bits: {bits_result}")
                else:
                    logger.warning(f"[INFERENCE] Too many classes ({len(classes)}) for 4-bit representation")

                consecutive_count = consecutive_count + 1 if predicted_class == previous_prediction else 1 
                if consecutive_count >= 3 and current_state != classes[predicted_class]:
                    self.write_queue.put(classes[predicted_class])
                    logger.info(f"[INFERENCE] Stable prediction: {classes[predicted_class]} (Confidence: {confidence:.2f})")
                    current_state = classes[predicted_class]
                    time.sleep(5)
                previous_prediction = predicted_class
                self.frame_queue.task_done()
                
            except queue.Empty:
                logger.debug("[INFERENCE] Frame queue empty, waiting...")
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"[INFERENCE] Exception: {e}")
                time.sleep(0.1)
                continue
                
            time.sleep(0.1)
    
    def start_all_threads(self, serial_comm, camera):
        """すべてのスレッドを開始"""
        # 基本スレッド
        self.threads = [
            threading.Thread(target=listen_serial, args=(self.stop_event, serial_comm, self.read_queue), daemon=True),
            threading.Thread(target=write_serial, args=(self.stop_event, serial_comm, self.write_queue), daemon=True),
            threading.Thread(target=self.distribute_frames, args=(camera,), daemon=True),
            threading.Thread(target=save_images, args=(
                self.stop_event, self.mode_train, self.frame_queue, self.image_queue, 
                camera, self.label_queue, self.start_time, self.config.directory.image_dir
            ), daemon=True),
            threading.Thread(target=self.process_images_with_bits, daemon=True),
            threading.Thread(target=self.handle_received_data, daemon=True),
            threading.Thread(target=self.user_input_listener, daemon=True)
        ]

        # 学習機能が有効な場合の追加スレッド
        if self.config.model.is_update:
            self.threads.extend([
                threading.Thread(target=monitor_folder, args=(
                    self.stop_event, self.start_train, self.config.directory.image_dir, 
                    self.config.directory.dataset_dir, self.config.monitoring.THRESHOLD, 
                    self.config.monitoring.CHECK_INTERVAL
                ), daemon=True),
                threading.Thread(target=train_controller, args=(
                    self.stop_event, self.start_train, self.config.hyperparameters.batch_size, 
                    self.config.hyperparameters.epochs, self.config.hyperparameters.img_size, 
                    self.config.hyperparameters.learning_rate, self.config.directory.dataset_dir, 
                    self.config.directory.model_dir, self.config.model.name, self.config.gpu.gpu_index, 
                    self.classes_queue, self.config.model.arch
                ), daemon=True)
            ])

        # すべてのスレッドを開始
        for thread in self.threads:
            thread.start()
            
        logger.info(f"[THREAD] Started {len(self.threads)} threads")
        return self.threads
    
    def wait_for_completion(self):
        """メインスレッド（user_input_listener）の終了を待つ"""
        self.threads[-1].join()  # user_input_listenerスレッドを待つ
        self.stop_event.set()
        
        # 他のすべてのスレッドの終了を待つ
        for thread in self.threads[:-1]:
            thread.join()
        
        logger.info("[THREAD] All threads completed")
    
    def cleanup(self, serial_comm, camera):
        """リソースのクリーンアップ"""
        serial_comm.close()
        camera.release()
        logger.info("[THREAD] Resources cleaned up") 