import pytest
import threading
import queue
from unittest.mock import MagicMock, patch
from centralmaneger.serial.communication import SerialCommunication
from main import main  # Import the main function

@pytest.fixture
def mock_serial():
    """Creates a mock object for SerialCommunication"""
    mock_serial = MagicMock(spec=SerialCommunication)
    mock_serial.read_serial.return_value = "S111100"  # Simulate received data
    return mock_serial

def test_main_quit(monkeypatch, capsys):
    """Test if entering 'q' or 'quit' stops the system correctly"""
    # Simulate user input: "q\n" (quit command)
    monkeypatch.setattr('sys.stdin', iter(["q\n"]))

    # Mock the SerialCommunication class to prevent real serial port access
    with patch("centralmaneger.serial.communication.SerialCommunication", autospec=True) as mock_serial_class:
        mock_serial_instance = mock_serial_class.return_value
        mock_serial_instance.read_serial.return_value = "S111100"  # Simulate receiving data
        mock_serial_instance.write_serial.return_value = None  # Simulate writing data

        main()  # Run the main function

    captured = capsys.readouterr()
    assert "[INFO] Stopping the system..." in captured.out
    assert "[INFO] System shut down successfully." in captured.out

def test_main_quit_alternative(monkeypatch, capsys):
    """Test if entering 'quit' also stops the system correctly"""
    # Simulate user input: "quit\n"
    monkeypatch.setattr('sys.stdin', iter(["quit\n"]))

    # Mock SerialCommunication to avoid real serial port access
    with patch("centralmaneger.serial.communication.SerialCommunication", autospec=True) as mock_serial_class:
        mock_serial_instance = mock_serial_class.return_value
        mock_serial_instance.read_serial.return_value = "S123456"  # Simulate receiving data
        mock_serial_instance.write_serial.return_value = None  # Simulate writing data

        main()  # Run the main function

    captured = capsys.readouterr()
    assert "[INFO] Stopping the system..." in captured.out
    assert "[INFO] System shut down successfully." in captured.out
