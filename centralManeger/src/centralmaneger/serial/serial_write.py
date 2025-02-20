import threading
import queue
import logging
from centralmaneger.serial.communication import SerialCommunication

# Logging setup
logger = logging.getLogger(__name__)

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
                logger.info(f"[WRITE] Sent data: C{command}")
                write_queue.task_done()
            except queue.Empty:
                pass  # Skip if the queue is empty
    except Exception as e:
        logger.exception(f"[ERROR] Serial writer encountered an error: {e}")

    logger.info("[INFO] Serial writer stopped gracefully.")
