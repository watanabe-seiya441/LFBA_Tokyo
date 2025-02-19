import threading
import queue
import time
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial
from centralmaneger.serial.serial_write import write_serial

# Serial port settings
SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 9600

# Thread management
stop_event = threading.Event()

# **Separate queues for received and sent data**
read_queue = queue.Queue()   # Queue for received data
write_queue = queue.Queue()  # Queue for data to be sent

def main():
    """
    Main thread: Manages serial listening and writing threads, 
    allows user input to be sent, and displays received data.
    """
    serial_comm = SerialCommunication(SERIAL_PORT, BAUDRATE)

    # Start the listening thread
    listen_thread = threading.Thread(target=listen_serial, args=(stop_event, serial_comm, read_queue), daemon=True)
    listen_thread.start()

    # Start the writing thread
    write_thread = threading.Thread(target=write_serial, args=(stop_event, serial_comm, write_queue), daemon=True)
    write_thread.start()

    print("[INFO] Serial communication started. Please enter data.")
    print("[INFO] Type 'q' or 'quit' to exit.")

    try:
        while True:
            # **Get user input**
            user_input = input("> ").strip()

            # **Exit system if "q" or "quit" is entered**
            if user_input.lower() in ["q", "quit"]:
                print("[INFO] Stopping the system...")
                break

            # **Send user input via serial**
            write_queue.put(user_input)

            # **Display received data if available**
            try:
                received_data = read_queue.get_nowait()
                print(f"[MAIN] Received data: {received_data}")
            except queue.Empty:
                pass  # Skip if no data is available

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Stopping the system...")

    # **Stop the threads**
    stop_event.set()

    # **Wait for threads to finish**
    listen_thread.join()
    write_thread.join()
    serial_comm.close()

    print("[INFO] System shut down successfully.")

if __name__ == "__main__":
    main()
