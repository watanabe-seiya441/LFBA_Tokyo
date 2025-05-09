# Central Manager

Central ManagerはLFBAシステムの中核を担うPCで、カメラからの画像入力、モデルの学習・推論、インフラへの制御信号出力を行います。本システムはPoetryで構築されています。

---

## インストール

### 1. Poetryのインストール

以下のコマンドを実行してPoetryをインストールします：

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. インストール確認

Poetryのバージョンを確認します：

```bash
poetry --version
```

期待される出力：

```
Poetry (version 2.1.1)
```

バージョンが表示されない場合は、PoetryをシステムPATHに追加してシェルを再読み込みします：

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 仮想環境の確認

Poetryの仮想環境のPython実行ファイルを確認します：

```bash
poetry run python -c "import sys; print(sys.executable)"
```

期待される出力例：

```
/home/kohki/.cache/pypoetry/virtualenvs/centralmanager-xA32UX7A-py3.12/bin/python
```

ホストのPythonが表示される場合は、仮想環境情報をチェックします：

```bash
poetry env info
```

`Virtualenv`の`Path`や`Executable`が`NA`であれば、以下でPythonバージョンを指定して修正します：

```bash
poetry env use python3.12
```

### 4. 依存関係のインストール

以下を実行して必要ライブラリをインストールします：

```bash
poetry install
```

---

## 設定

プロジェクトのルートに`config.toml`を作成して、以下の構成を記述します：

```toml
[serial]
port = "/dev/ttyACM0"
baudrate = 9600

[camera]
cameraID = 0

[directory]
image_dir = "image/recorded"
dataset_dir = "dataset"
model_dir = "model"

[model]
classes = ["0000", "0001", "0010", "0011"]
name = "mobilenetv3_small_latest.pth"
arch = "mobilenet"
is_update = false

[monitoring]
THRESHOLD = 10000  # ファイル数の閾値
CHECK_INTERVAL = 60

[hyperparameters]
batch_size = 32
epochs = 5
img_size = 224
learning_rate = 0.001

[gpu]
gpu_index = 1
``` 

### 設定項目の詳細

- **[serial]**: シリアル通信設定
  - `port`: 接続ポートを指定
  - `baudrate`: 通信速度（デフォルト9600）

- **[camera]**: カメラIDを設定

- **[directory]**: 画像、データセット、モデル保存用ディレクトリ

- **[model]**:
  - `classes`: 制御信号クラス
  - `name`: モデルファイル名（.pth）
  - `arch`: モデルアーキテクチャ（mobilenetまたはvgg）
  - `is_update`: 継続学習を行うか

- **[monitoring]**:
  - `THRESHOLD`: 学習開始前の画像数
  - `CHECK_INTERVAL`: 監視間隔（秒）

- **[hyperparameters]**: モデル学習パラメータ

- **[gpu]**: GPUインデックス（単一GPUなら0）

---

## 使い方

1. カメラとスイッチをPCに接続
2. 以下を実行してシステムを起動します：

    ```bash
    cd centralManager/
    poetry run python main.py
    ```

3. 終了するにはターミナルで`q`または`quit`を入力

---

## ディレクトリとログ

- **log/**: ログファイル保存先。問題発生時は内容を確認してください。
- **image/**: キャプチャ画像保存先。`train`モード時のみ、スイッチ入力変更後15～180秒で撮影されます。

---

以上がCentral Managerの導入および運用手順です。詳細はログを参照するか、管理者までお問い合わせください。
