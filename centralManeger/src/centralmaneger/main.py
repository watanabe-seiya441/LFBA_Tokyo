import time
from centralmaneger.serial.communication import SerialCommunication
from centralmaneger.serial.serial_reader import listen_serial


def main():
    # シリアルポートとボーレートを適宜設定
    port = "/dev/ttyUSB0"  # 使用するポートを指定
    baudrate = 9600  # 適切なボーレートを設定

    try:
        serial_comm = SerialCommunication(port, baudrate)
        print("Serial communication started.")

        # シリアル通信のリスニングを開始
        listen_serial(serial_comm)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        serial_comm.close()
        print("Serial communication closed.")


if __name__ == "__main__":
    main()