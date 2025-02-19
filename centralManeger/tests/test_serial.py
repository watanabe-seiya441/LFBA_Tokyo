import pytest
import threading
import queue
from unittest.mock import MagicMock, patch
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial
from centralmaneger.serial.serial_write import write_serial

@pytest.fixture
def mock_serial():
    """SerialCommunication のモックオブジェクトを作成"""
    mock_serial = MagicMock(spec=SerialCommunication)
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
    mock_serial.read_serial.side_effect = ["S111100", "S001101", "S1111111", "C110000", None]  # 受信データをシミュレート

    listen_thread = threading.Thread(target=listen_serial, args=(stop_event, mock_serial, read_queue))
    listen_thread.start()

    threading.Event().wait(0.1)
    stop_event.set()
    listen_thread.join()

    assert read_queue.get() == "S111100"
    assert read_queue.get() == "S001101"

    captured = capsys.readouterr()
    assert "[READ] Received data: S111100" in captured.out
    assert "[READ] Received data: S001101" in captured.out
    assert "[READ] Received data: S1111111" not in captured.out
    assert "[READ] Received data: C110000" not in captured.out

def test_write_serial(mock_serial, write_queue, capsys):
    """ライトスレッドのテスト"""
    stop_event = threading.Event()

    write_queue.put("1111")
    write_queue.put("0010")

    write_thread = threading.Thread(target=write_serial, args=(stop_event, mock_serial, write_queue))
    write_thread.start()

    threading.Event().wait(0.1)
    stop_event.set()
    write_thread.join()

    mock_serial.write_serial.assert_any_call("C1111")
    mock_serial.write_serial.assert_any_call("C0010")

    captured = capsys.readouterr()
    assert "[WRITE] Sent data: C1111" in captured.out
    assert "[WRITE] Sent data: C0010" in captured.out
