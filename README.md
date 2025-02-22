# LFBA_Thailand

This repository is intended for the study of LFBA at Chulalongkorn University.

## Installation

This repository contains two directories: `"switcher"` and `"centralManager"`.  
Please download this repository to your control PC to use LFBA.

### Switcher

The `"switcher"` directory contains the code for the Arduino-based wall switch.  
To use it, upload the `switch.ino` file to the Arduino using the Arduino IDE.

### CentralManager

The `"centralManager"` directory contains the control PC software for LFBA.  
We use **Poetry** for dependency management, so please install Poetry first.

To set up the environment, run the following commands:

```sh
cd centralManager/
poetry install
```
Now, your control PC is ready to run LFBA.

Usage
To start the system, follow these steps:

Ensure that your camera and wall switch are connected to the control PC.
Run the following command:
```sh
cd centralManager/
poetry run python main.py
```
To stop the system, enter `q` or `quit` in the terminal.

------メモ ---------
## Switcherの仕様
#### Manualモード
- ボタンあるいはトグルスイッチが押されるとsignalを発信
    - S {ボタン4つ} {forgetボタン} {トグルスイッチ2つ}
#### Autoモード
- Main PCからCから始まるコマンドが来るとそれの通りにボタンを操作
    - C {ボタン4つ}

ここからPCの仕様として必要なものは以下の通り
- Sから始まるSignalを常に受け取るようにする
- 受け取ったら
    - ボタン4つの情報を保存
    - is_Recordならカメラを操作
- 逆に推論した場合、Cと4桁の数字を出す。