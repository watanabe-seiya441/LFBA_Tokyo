# centralManeger/src/centralmaneger/serial/serial_reader.py
from centralmaneger.serial.communication import SerialCommunication
import re


def listen_serial(serial_communication: SerialCommunication) -> None:
    """
    Reads data from the serial port.

    Args:
        serial_communication (SerialCommunication): The serial communication object.

    Returns:
        str: Data read from the serial port, stripped of whitespace.
             Returns an empty string if no data is available.
    """
    while True:        
        data = serial_communication.read_serial()
        pattern = r"^S\d{7}"
        if data and re.match(pattern, data):
            print(f"Received data: {data}")
