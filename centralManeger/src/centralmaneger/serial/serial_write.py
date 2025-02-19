import threading
from centralmaneger.serial.communication import SerialCommunication
import queue

def write_serial(stop_event: threading.Event, serial_communication: SerialCommunication, write_queue: queue.Queue) -> None:
    """
    Retrieves data from the write_queue and sends it to the serial port.

    Args:
        stop_event (threading.Event): Event flag to stop the thread
        serial_communication (SerialCommunication): Serial communication object
        write_queue (queue.Queue): Queue containing data to be sent
    """
    try:
        while not stop_event.is_set():
            try:
                command = write_queue.get(timeout=1)  # **Retrieve data from the queue**
                serial_communication.write_serial(f"C{command}")  # **Send data via serial**
                print(f"[WRITE] Sent data: C{command}")
                write_queue.task_done()
            except queue.Empty:
                pass  # Skip if the queue is empty
    except Exception as e:
        print(f"[ERROR] Serial writer error: {e}")

    print("[INFO] Serial writer stopped.")
