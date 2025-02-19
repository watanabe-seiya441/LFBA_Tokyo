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

def save_images(stop_event, frame_queue, image_queue):
    """
    Saves images from the latest frame queue at a 1-second interval.

    Args:
        stop_event (threading.Event): Event flag to stop the thread.
        frame_queue (queue.Queue): Queue to fetch the latest frame.
        image_queue (queue.Queue): Queue to store captured image filenames.
    """
    while not stop_event.is_set():
        start_time = time.time()

        if frame_queue.empty():
            time.sleep(0.1)
            continue  # Skip if no frame is available

        frame = frame_queue.get()

        # Save the image every 1 second
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        image_queue.put(filename)
        print(f"[INFO] Image saved: {filename}")

        # Ensure the next capture happens after 1 second
        elapsed_time = time.time() - start_time
        sleep_time = max(0, 1 - elapsed_time)
        time.sleep(sleep_time)

def main():
    """
    Initializes and starts the camera application.
    """
    camera = Camera(camera_id=0, capture_interval=1)
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=1)
    image_queue = queue.Queue()

    # Start frame capture thread
    capture_thread = threading.Thread(target=capture_latest_frame, args=(camera, frame_queue, stop_event), daemon=True)
    capture_thread.start()

    # Start image saving thread
    save_thread = threading.Thread(target=save_images, args=(stop_event, frame_queue, image_queue), daemon=True)
    save_thread.start()

    try:
        while not stop_event.is_set():
            if not frame_queue.empty():
                frame = frame_queue.get()
                cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF  # Wait for key input
            if key == ord('q'):
                print("[INFO] Stopping the camera...")
                stop_event.set()  # Signal the threads to stop
                break

            time.sleep(0.1)  # Prevent high CPU usage

    finally:
        capture_thread.join()  # Ensure the capture thread has stopped
        save_thread.join()  # Ensure the saving thread has stopped
        camera.release()
        print("[INFO] Camera application exited cleanly.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
