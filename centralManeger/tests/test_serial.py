import pytest
import threading
import queue
from unittest.mock import MagicMock
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial
from centralmaneger.serial.serial_write import write_serial
from main import main

@pytest.fixture
def mock_serial():
    """SerialCommunication のモックオブジェクトを作成"""
    mock_serial = MagicMock(spec=SerialCommunication)
    mock_serial.read_serial.side_effect = ["S123456", "S765432", "", None]  # **受信データをシミュレート**
    return mock_serial

@pytest.fixture
def read_queue():
    """受信データ用のキューを作成"""
    return queue.Queue()

@pytest.fixture
def write_queue():
    """送信データ用のキューを作成"""
    return queue.Queue()

def test_listen_serial(mock_serial, read_queue, capsys):
    """リッスンスレッドのテスト"""
    stop_event = threading.Event()
    listen_thread = threading.Thread(target=listen_serial, args=(stop_event, mock_serial, read_queue))
    listen_thread.start()

    threading.Event().wait(0.1)
    stop_event.set()
    listen_thread.join()

    assert read_queue.get() == "S123456"
    assert read_queue.get() == "S765432"

    captured = capsys.readouterr()
    assert "[READ] Received data: S123456" in captured.out
    assert "[READ] Received data: S765432" in captured.out

def test_write_serial(mock_serial, write_queue, capsys):
    """ライトスレッドのテスト"""
    stop_event = threading.Event()

    write_queue.put("ACK123456")
    write_queue.put("ACK765432")

    write_thread = threading.Thread(target=write_serial, args=(stop_event, mock_serial, write_queue))
    write_thread.start()

    threading.Event().wait(0.1)
    stop_event.set()
    write_thread.join()

    mock_serial.write_serial.assert_any_call("ACK123456")
    mock_serial.write_serial.assert_any_call("ACK765432")

    captured = capsys.readouterr()
    assert "[WRITE] Sent data: ACK123456" in captured.out
    assert "[WRITE] Sent data: ACK765432" in captured.out
