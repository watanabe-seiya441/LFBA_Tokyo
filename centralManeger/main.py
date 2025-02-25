import threading
import queue
import time
import logging
import os
from datetime import datetime
from centralmaneger.camera.camera import Camera
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial
from centralmaneger.serial.serial_write import write_serial
from centralmaneger.camera.cameraApp import capture_latest_frame, save_images

# Serial port settings
SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 9600

# Thread management
stop_event = threading.Event()

# Separate queues for received and sent data
read_queue = queue.Queue()   # Queue for received data
write_queue = queue.Queue()  # Queue for data to be sent

# Queues for camera
frame_queue = queue.Queue(maxsize=1)
image_queue = queue.Queue()
label_queue = queue.Queue(maxsize=1)  # Queue to store the latest received data

start_time = datetime.now().strftime("%Y%m%dT%H%M%S")

# Logging setup
log_dir = "log"
os.makedirs(log_dir, exist_ok=True) 
log_filename = f"{log_dir}/system_{start_time}.log"
logging.basicConfig(
    filename=log_filename, 
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def handle_received_data(stop_event: threading.Event, read_queue: queue.Queue, label_queue: queue.Queue) -> None:
    """
    Thread function to process received serial data and update label queue.
    """
    latest_received_data = "unknown"
    while not stop_event.is_set():
        try:
            received_data = read_queue.get(timeout=0.1)
            latest_received_data = received_data[1:5] 
            if not label_queue.empty():
                label_queue.get()  # Clear old label
            if received_data[6] == '0': # If it is currently in manual mode
                label_queue.put(latest_received_data)
        except queue.Empty:
            pass  # No new data, continue loop

def user_input_listener(stop_event: threading.Event, write_queue: queue.Queue) -> None:
    """
    Thread function to handle user input separately from received data processing.
    """
    print("[INFO] Serial communication and camera started. Please enter data.")
    print("[INFO] Type 'q' or 'quit' to exit.")
    
    try:
        while not stop_event.is_set():
            user_input = input("> ").strip()
            
            if user_input.lower() in ["q", "quit"]:
                print("[INFO] Stopping the system...")
                stop_event.set()
                break
            
            write_queue.put(user_input)
    except KeyboardInterrupt:
        logger.warning("[INFO] Interrupted by user. Stopping the system...")
        stop_event.set()

def main():
    """
    Main function to manage serial listening, writing, camera processing, and user input handling.
    """
    print(f"[INFO] System started at {start_time}")

    serial_comm = SerialCommunication(SERIAL_PORT, BAUDRATE)
    camera = Camera(camera_id=0, capture_interval=1)

    # Start the listening thread
    listen_thread = threading.Thread(target=listen_serial, args=(stop_event, serial_comm, read_queue), daemon=True)
    listen_thread.start()

    # Start the writing thread
    write_thread = threading.Thread(target=write_serial, args=(stop_event, serial_comm, write_queue), daemon=True)
    write_thread.start()

    # Start the camera processing threads
    capture_thread = threading.Thread(target=capture_latest_frame, args=(camera, frame_queue, stop_event), daemon=True)
    capture_thread.start()
    
    save_thread = threading.Thread(target=save_images, args=(stop_event, frame_queue, image_queue, camera, label_queue, start_time), daemon=True)
    save_thread.start()

    # Start received data processing thread
    receiver_thread = threading.Thread(target=handle_received_data, args=(stop_event, read_queue, label_queue), daemon=True)
    receiver_thread.start()

    # Start user input listener thread
    input_thread = threading.Thread(target=user_input_listener, args=(stop_event, write_queue), daemon=True)
    input_thread.start()
    
    # Wait for threads to finish
    input_thread.join()
    stop_event.set()
    listen_thread.join()
    write_thread.join()
    capture_thread.join()
    save_thread.join()
    receiver_thread.join()
    serial_comm.close()
    camera.release()

    print("[INFO] System shut down successfully.")

if __name__ == "__main__":
    main()
