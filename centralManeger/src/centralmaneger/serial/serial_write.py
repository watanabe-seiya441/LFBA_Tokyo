import threading
from centralmaneger.serial.communication import SerialCommunication
import queue

def write_serial(stop_event: threading.Event, serial_communication: SerialCommunication, write_queue: queue.Queue) -> None:
    """
    write_queue からデータを取得し、シリアルポートに送信する。

    Args:
        stop_event (threading.Event): スレッドの停止フラグ
        serial_communication (SerialCommunication): シリアル通信オブジェクト
        write_queue (queue.Queue): 送信データを格納するキュー
    """
    try:
        while not stop_event.is_set():
            try:
                data_to_send = write_queue.get(timeout=1)  # **送信データを取得**
                serial_communication.write_serial(data_to_send)  # **シリアル送信**
                print(f"[WRITE] Sent data: {data_to_send}")
                write_queue.task_done()
            except queue.Empty:
                pass  # キューが空ならスキップ
    except Exception as e:
        print(f"[ERROR] Serial writer error: {e}")

    print("[INFO] Serial writer stopped.")