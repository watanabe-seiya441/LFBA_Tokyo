import time
import threading
import queue
from datetime import datetime
from centralmaneger.camera.camera import Camera

def capture_latest_frame(camera: Camera, frame_queue: queue.Queue, stop_event: threading.Event) -> None:
    """
    Continuously captures the latest frame from the camera and updates the frame queue.

    Args:
        camera (Camera): An instance of the Camera class.
        frame_queue (queue.Queue): Queue to store the latest frame.
        stop_event (threading.Event): Event flag to stop the thread.
    """
    while not stop_event.is_set():
        frame = camera.capture_frame()
        if frame is not None:
            if not frame_queue.empty():
                frame_queue.get()  # Remove the old frame
            frame_queue.put(frame)

def save_images(stop_event: threading.Event, frame_queue: queue.Queue, image_queue: queue.Queue, camera: Camera, name_queue: queue.Queue) -> None:
    """
    Saves images from the latest frame queue at a 1-second interval, including received data in the filename.

    Args:
        stop_event (threading.Event): Event flag to stop the thread.
        frame_queue (queue.Queue): Queue to fetch the latest frame.
        image_queue (queue.Queue): Queue to store captured image filenames.
        camera (Camera): An instance of the Camera class to handle image saving.
        name_queue (queue.Queue): Queue to store the latest received data for filename.
    """
    latest_received_data = "unknown"  # Default name if no data received

    while not stop_event.is_set():
        start_time = time.time()

        if frame_queue.empty():
            time.sleep(0.1)
            continue  # Skip if no frame is available

        frame = frame_queue.get()

        # Get the latest received data if available
        if not name_queue.empty():
            latest_received_data = name_queue.get()

        # Save the image with received data in the filename
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"{timestamp}_{latest_received_data}.jpg"
        camera.save_image(filename, frame)
        image_queue.put(filename)
        print(f"[INFO] Image saved: {filename}")

        # Ensure the next capture happens after 1 second
        elapsed_time = time.time() - start_time
        sleep_time = max(0, 1 - elapsed_time)
        time.sleep(sleep_time)
