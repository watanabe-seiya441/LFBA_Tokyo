# centralManeger/src/centralmaneger/serial/serial_reader.py
from centralmaneger.serial.communication import SerialCommunication
import re
import threading
import queue

def listen_serial(stop_event: threading.Event, serial_communication: SerialCommunication, read_queue: queue.Queue) -> None:
    """
    シリアルポートからデータを読み取り、read_queue に格納する。

    Args:
        stop_event (threading.Event): スレッドの停止フラグ
        serial_communication (SerialCommunication): シリアル通信オブジェクト
        read_queue (queue.Queue): 受信データを格納するキュー
    """
    try:
        while not stop_event.is_set():
            data = serial_communication.read_serial()
            pattern = r"^S\d{6}$"
            if data and re.match(pattern, data):
                print(f"[READ] Received data: {data}")
                read_queue.put(data)  # **受信データを read_queue に格納**
    except StopIteration:
        print("[ERROR] Serial listener stopped due to StopIteration.")

    print("[INFO] Serial listener stopped gracefully.")
