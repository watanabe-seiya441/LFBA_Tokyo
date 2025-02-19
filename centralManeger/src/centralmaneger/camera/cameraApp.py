import cv2
import time
import threading
import queue
from datetime import datetime
from centralmaneger.camera.camera import Camera

def capture_latest_frame(camera, frame_queue, stop_event):
    """
    Continuously captures the latest frame from the camera and updates the frame queue.

    Args:
        camera (Camera): An instance of the Camera class.
        frame_queue (queue.Queue): Queue to store the latest frame.
        stop_event (threading.Event): Event flag to stop the thread.
    """
    while not stop_event.is_set():
        ret, frame = camera.cap.read()
        if ret:
            if not frame_queue.empty():
                frame_queue.get()  # Remove the old frame
            frame_queue.put(frame)

def run_camera_app(stop_event, camera, image_queue):
    """
    Runs the camera application in a separate thread.

    Args:
        stop_event (threading.Event): Event flag to stop the thread.
        camera (Camera): An instance of the Camera class.
        image_queue (queue.Queue): Queue to store captured image filenames.
    """
    frame_queue = queue.Queue(maxsize=1)  # Store only the latest frame
    frame_thread = threading.Thread(target=capture_latest_frame, args=(camera, frame_queue, stop_event), daemon=True)
    frame_thread.start()

    while not stop_event.is_set():
        start_time = time.time()

        if frame_queue.empty():
            continue  # Skip if no frame is available

        frame = frame_queue.get()
        cv2.imshow("Camera", frame)

        # Save the image every 1 second
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"{timestamp}.jpg"
        camera.save_image(filename, frame)
        image_queue.put(filename)

        # Wait until 1 second has passed
        elapsed_time = time.time() - start_time
        sleep_time = max(0, camera.capture_interval - elapsed_time)
        time.sleep(sleep_time)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break

    camera.release()
    print("[INFO] Camera thread stopped gracefully.")

def main():
    """
    Initializes and starts the camera application.
    """
    camera = Camera(camera_id=0, capture_interval=1)
    stop_event = threading.Event()
    image_queue = queue.Queue()

    # Start camera thread
    camera_thread = threading.Thread(target=run_camera_app, args=(stop_event, camera, image_queue), daemon=True)
    camera_thread.start()

    try:
        while not stop_event.is_set():
            key = cv2.waitKey(1) & 0xFF  # Wait for key input
            if key == ord('q'):
                print("[INFO] Stopping the camera...")
                stop_event.set()  # Signal the camera thread to stop
                break
            time.sleep(0.1)  # Prevent high CPU usage

    finally:
        camera_thread.join()  # Ensure the camera thread has stopped
        camera.release()
        print("[INFO] Camera application exited cleanly.")

if __name__ == "__main__":
    main()
