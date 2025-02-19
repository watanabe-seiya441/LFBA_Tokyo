from centralmaneger.serial.communication import SerialCommunication
import re
import threading
import queue

def listen_serial(stop_event: threading.Event, serial_communication: SerialCommunication, read_queue: queue.Queue) -> None:
    """
    Reads data from the serial port and stores it in the read_queue.

    Args:
        stop_event (threading.Event): Event flag to stop the thread
        serial_communication (SerialCommunication): Serial communication object
        read_queue (queue.Queue): Queue to store received data
    """
    try:
        while not stop_event.is_set():
            data = serial_communication.read_serial()
            pattern = r"^S\d{6}$"
            if data and re.match(pattern, data):
                print(f"[READ] Received data: {data}")
                read_queue.put(data)  # **Store received data in read_queue**
    except StopIteration:
        print("[ERROR] Serial listener stopped due to StopIteration.")

    print("[INFO] Serial listener stopped gracefully.")
