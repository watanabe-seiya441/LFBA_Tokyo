# Central Manager

The Central Manager is the core PC of the LFBA system. It handles image input from cameras, model training and inference, and outputs control signals to the infrastructure. This system is built using Poetry.

---

## Installation

### 1. Install Poetry

Run the following command to install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Verify Installation

Check the installed Poetry version:

```bash
poetry --version
```

Expected output:

```
Poetry (version 2.1.1)
```

If no version is displayed, add Poetry to your system PATH and reload the shell:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Check Virtual Environment

Verify the virtual environment's Python executable:

```bash
poetry run python -c "import sys; print(sys.executable)"
```

Expected output:

```
/home/kohki/.cache/pypoetry/virtualenvs/centralmanager-xA32UX7A-py3.12/bin/python
```

If the host Python is displayed instead, check the virtual environment details:

```bash
poetry env info
```

If the `Path` and `Executable` fields under `Virtualenv` are `NA`, fix it by specifying the correct Python version:

```bash
poetry env use python3.12
```

### 4. Install Dependencies

Run the following command to install required libraries:

```bash
poetry install
```

---

## Configuration

Create a `config.toml` file with the following structure:

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
# name = "vgg_best_model.pth"
arch = "mobilenet"
# arch = "vgg"

[monitoring]
THRESHOLD = 10000  # File count threshold
CHECK_INTERVAL = 60

[hyperparameters]
batch_size = 32
epochs = 5
img_size = 224
learning_rate = 0.001

[gpu]
gpu_index = 1
```

### Configuration Details

- **[serial]**: Set serial communication parameters.
  - `port`: Specify the connected port.
  - `baudrate`: Use `9600` as the default.

- **[camera]**: Set the camera ID.

- **[directory]**: Specify directories for saving images, datasets, and models. If an existing model is used, save the `.pth` file in the model directory.

- **[model]**:
  - `classes`: Define control signal classes.
  - `name`: Specify the model file name.
  - `arch`: Set the model architecture. Currently supports `mobilenet` and `vgg`.

- **[monitoring]**:
  - `THRESHOLD`: Number of images before starting training.
  - `CHECK_INTERVAL`: Monitoring interval in seconds.

- **[hyperparameters]**: Define model training parameters.

- **[gpu]**: Specify GPU index. Use `0` if only one GPU is available.

---

## Usage

1. Ensure that the camera and switch are connected to the PC.
2. Run the following command to start the system:

```bash
cd centralManager/
poetry run python main.py
```

3. To stop the system, enter `q` or `quit` in the terminal.

---

## Directories and Logs

- **Log Directory**: Contains log files. If issues occur, check the logs for details.
- **Image Directory**: Stores captured images. Images are captured only in `train` mode, and only after 15 to 180 seconds from a switch input change.

---

This document provides a clear guide for setting up and operating the Central Manager. For any further assistance, please consult the logs or contact the system administrator.

# Central Manager

This project is designed to manage various operations, including camera control, dataset creation, image classification, model training, and serial communication. The following is an overview of the directory structure and its components.

## Directory Structure

```
centralManeger
├── README.md
├── config.toml
├── dataset
├── image
├── log
├── main.py
├── model
├── poetry.lock
├── pyproject.toml
├── src
│   └── centralmaneger
│       ├── __init__.py
│       ├── camera
│       │   ├── __init__.py
│       │   ├── camera.py
│       │   └── cameraApp.py
│       ├── create_dataset
│       │   ├── __init__.py
│       │   └── folder_monitor.py
│       ├── image_classification
│       │   ├── __init__.py
│       │   └── classifier.py
│       ├── model_training
│       │   ├── __init__.py
│       │   └── model_training.py
│       └── serial
│           ├── __init__.py
│           ├── communication.py
│           ├── serial_reader.py
│           └── serial_write.py
└── tests
    ├── __init__.py
    ├── test_main.py
    └── test_serial.py
```

## Directory Descriptions

### Root Level
- **`README.md`**  
  Provides an overview and documentation for the project.

- **`config.toml`**  
  Contains configuration settings for the project.

- **`dataset`**  
  Stores datasets used for model training and evaluation.

- **`image`**  
  Holds images captured or used during processing.

- **`log`**  
  Contains log files generated during the program's execution.

- **`main.py`**  
  The main entry point of the application.

- **`model`**  
  Stores trained machine learning models.

- **`poetry.lock`**  
  Locks dependencies to specific versions for consistent environments.

- **`pyproject.toml`**  
  Defines project metadata and dependencies for Python's Poetry package manager.

### `src/centralmaneger`  
This is the core source directory containing modules for different functionalities.

- **`__init__.py`**  
  Initializes the `centralmaneger` package.

#### `camera`  
Handles camera operations.
- **`camera.py`**  
  Contains the core logic for interfacing with the camera hardware.
- **`cameraApp.py`**  
  Provides application-level functionality using the camera module.

#### `create_dataset`  
Manages the creation and organization of datasets.
- **`folder_monitor.py`**  
  Monitors directories and manages dataset creation dynamically.

#### `image_classification`  
Responsible for image classification tasks.
- **`classifier.py`**  
  Contains the logic for classifying images using machine learning models.

#### `model_training`  
Handles training processes for machine learning models.
- **`model_training.py`**  
  Implements the model training pipeline.

#### `serial`  
Manages serial communication with external devices.
- **`communication.py`**  
  Provides functions for managing serial communication.
- **`serial_reader.py`**  
  Handles reading data from serial ports.
- **`serial_write.py`**  
  Manages writing data to serial ports.

### `tests`  
Contains test files to ensure the stability and correctness of the code. (But not working...)

- **`__init__.py`**  
  Initializes the `tests` package.

- **`test_main.py`**  
  Contains unit tests for the `main.py` module.

- **`test_serial.py`**  
  Contains unit tests for serial communication modules.

---

## ⚙️ Installation

```
poetry install
```

---

## 🚀 Usage

```
poetry run python main.py
```

---

## 🧪 Running Tests

```
poetry run pytest
```

---

## 📄 License

This project is licensed under the MIT License.

