# centralManeger/src/centralmaneger/camera/cameraApp.py
import time
import threading
import queue
import os
import logging
from datetime import datetime
from centralmaneger.camera.camera import Camera

# Logging setup (グローバルな logger を定義)
logger = logging.getLogger(__name__)

def capture_latest_frame(camera: Camera, frame_queue: queue.Queue, stop_event: threading.Event, mode_record: threading.Event) -> None:
    """
    Continuously captures the latest frame from the camera and updates the frame queue.

    Args:
        camera (Camera): An instance of the Camera class.
        frame_queue (queue.Queue): Queue to store the latest frame.
        stop_event (threading.Event): Event flag to stop the thread.
    """
    while not stop_event.is_set():
        if not mode_record.is_set():
            continue
            
        frame = camera.capture_frame()
        if frame is not None:
            if not frame_queue.empty():
                frame_queue.get()  # Remove the old frame
            frame_queue.put(frame)
        logger.debug("[CAPTURE] Frame captured.")

def should_save_image(current_time: float, last_label_update_time: float, label: str, mode_train: threading.Event) -> bool:
    """Determine if an image should be saved based on time constraints and training mode."""
    return (
        last_label_update_time 
        and 15 <= (current_time - last_label_update_time) <= 180 
        and label != "unknown" 
        and mode_train.is_set()
    )

def save_image(camera, frame, start_time, label, image_queue, IMAGE_DIR):
    """Save an image with the appropriate filename and store it in the image queue."""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    # dir_path = f"image/{start_time}"
    os.makedirs(IMAGE_DIR, exist_ok=True)
    filename = f"{IMAGE_DIR}/{timestamp}_{label}.jpg"
    
    camera.save_image(filename, frame)
    image_queue.put(filename)
    logger.info(f"[SAVE] Image saved: {filename}")

def save_images(
    stop_event: threading.Event, mode_train: threading.Event, 
    frame_queue: queue.Queue, image_queue: queue.Queue, 
    camera, label_queue: queue.Queue, start_time: str, IMAGE_DIR: str
) -> None:
    """Saves images at a 1-second interval based on training mode and label updates."""
    
    latest_label = "unknown"
    last_label_update_time = None

    while not stop_event.is_set():
        current_time = time.time()

        if frame_queue.empty():
            time.sleep(0.1)
            continue

        frame = frame_queue.get()

        # Get the latest label if available
        try:
            latest_label = label_queue.get_nowait()
            last_label_update_time = time.time()
        except queue.Empty:
            pass  

        # Save the image if conditions are met
        if should_save_image(current_time, last_label_update_time, latest_label, mode_train):
            save_image(camera, frame, start_time, latest_label, image_queue, IMAGE_DIR)

        # Ensure a 1-second interval between captures
        time.sleep(max(0, 1 - (time.time() - current_time)))
