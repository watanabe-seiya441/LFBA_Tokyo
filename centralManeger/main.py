import threading
import queue
import time
from datetime import datetime
from centralmaneger.camera.camera import Camera
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial
from centralmaneger.serial.serial_write import write_serial
from centralmaneger.camera.cameraApp import capture_latest_frame, save_images
from centralmaneger.camera.camera import Camera

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
name_queue = queue.Queue(maxsize=1)  # Queue to store the latest received data

def main():
    """
    Main thread: Manages serial listening, writing, and camera processing threads,
    allows user input to be sent via serial, and displays received data.
    """
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
    
    save_thread = threading.Thread(target=save_images, args=(stop_event, frame_queue, image_queue, camera, name_queue), daemon=True)
    save_thread.start()

    print("[INFO] Serial communication and camera started. Please enter data.")
    print("[INFO] Type 'q' or 'quit' to exit.")

    try:
        latest_received_data = "unknown"
        while True:
            # Get user input
            user_input = input("> ").strip()

            # Exit system if "q" or "quit" is entered
            if user_input.lower() in ["q", "quit"]:
                print("[INFO] Stopping the system...")
                break

            # Send user input via serial
            write_queue.put(user_input)

            # Display received data if available
            try:
                received_data = read_queue.get_nowait()
                latest_received_data = received_data  # Update latest received data
                if not name_queue.empty():
                    name_queue.get()  # Clear old name
                name_queue.put(latest_received_data)
                print(f"[MAIN] Received data: {received_data}")
            except queue.Empty:
                pass  # Skip if no data is available

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Stopping the system...")

    # Stop the threads
    stop_event.set()

    # Wait for threads to finish
    listen_thread.join()
    write_thread.join()
    capture_thread.join()
    save_thread.join()
    serial_comm.close()
    camera.release()

    print("[INFO] System shut down successfully.")

if __name__ == "__main__":
    main()
