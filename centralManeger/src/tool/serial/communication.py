# centralManeger/src/centralmaneger/serial/communication.py
import serial

class SerialCommunication:
    def __init__(self, port: str, baudrate: int) -> None:
        """
        Initializes SerialCommunication with specified port and baudrate.

        Args:
            port (str): The serial port identifier.
            baudrate (int): The baudrate for serial communication.
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(self.port, self.baudrate)

    def read_serial(self) -> str:
        """
        Reads data from the serial port.

        Returns:
            str: Data read from the serial port, stripped of whitespace.
                 Returns an empty string if no data is available.
        """
        if self.ser and self.ser.in_waiting > 0:
            read_data = self.ser.readline().decode().strip()
            return read_data

    def write_serial(self, data: str) -> None:
        """
        Writes data to the serial port.

        Args:
            data (str): The data to write.
        """
        if self.ser:
            self.ser.write(data.encode()+ b"\r\n")

    def close(self) -> None:
        """
        Closes the serial connection.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
