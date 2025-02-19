import threading
import queue
import time
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial
from centralmaneger.serial.serial_write import write_serial

# シリアルポート設定
SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 9600

# スレッド管理用
stop_event = threading.Event()

# **受信データと送信データのキューを独立**
read_queue = queue.Queue()   # リッスンデータの格納
write_queue = queue.Queue()  # 送信データの格納

def main():
    """
    メインスレッド: シリアルリッスン・シリアルライトのスレッドを管理し、受信データを処理
    """
    serial_comm = SerialCommunication(SERIAL_PORT, BAUDRATE)

    # リッスンスレッド開始
    listen_thread = threading.Thread(target=listen_serial, args=(stop_event, serial_comm, read_queue))
    listen_thread.start()

    # ライトスレッド開始
    write_thread = threading.Thread(target=write_serial, args=(stop_event, serial_comm, write_queue))
    write_thread.start()

    # メインスレッドで受信データを処理
    try:
        for _ in range(20):  # 10秒間ループ（1回 0.5 秒 x 20回）
            time.sleep(0.5)
            try:
                received_data = read_queue.get_nowait()  # **受信データを取得**
                print(f"[MAIN] Processing received data: {received_data}")

                # **受信データに応じたレスポンスを送信**
                if received_data == "S123456":
                    write_queue.put("ACK123456")
                elif received_data == "S765432":
                    write_queue.put("ACK765432")
                else:
                    write_queue.put("UNKNOWN")
                
            except queue.Empty:
                pass  # 受信データがない場合はスキップ

    except KeyboardInterrupt:
        print("[INFO] Program interrupted by user.")

    # 10秒後にスレッドを停止
    stop_event.set()

    # スレッドの終了を待機
    listen_thread.join()
    write_thread.join()
    serial_comm.close()

    print("[INFO] Main program stopped.")

if __name__ == "__main__":
    main()
