# centralManeger/src/centralmaneger/serial/serial_reader.py
from centralmaneger.serial.communication import SerialCommunication
import re
import threading
import queue
import logging

# Logging setup
logger = logging.getLogger(__name__)

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
                logger.info(f"[READ] Received data: {data}")
                read_queue.put(data)  # **Store received data in read_queue**
    except StopIteration:
        logger.error("[ERROR] Serial listener stopped due to StopIteration.")
    except Exception as e:
        logger.exception(f"[ERROR] Exception in listen_serial: {e}")

    logger.info("[INFO] Serial listener stopped gracefully.")
