# Central Maneger
The Central Manager is the core PC of the LFBA system, responsible for receiving image inputs from cameras, training and inference of the model, and outputting control signals to the infrastructure.  
You can build using Poetry.

## Install
Install Poetry using the official installer:
```sh
$ curl -sSL https://install.python-poetry.org | python3 -
```
Check the installed version:
```sh
$ poetry --version
Poetry (version 2.1.1)
```
If no version is displayed:
Add Poetry to your system PATH and reload the shell:
```sh
$ echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
$ source ~/.bashrc
```
Verify the virtual environment:
```sh
$ poetry run python -c "import sys; print(sys.executable)"
/home/kohki/.cache/pypoetry/virtualenvs/centralmanager-xA32UX7A-py3.12/bin/python
```
If the output shows the host Python instead of the virtual environment's Python, the virtual environment may not be working correctly.

Check virtual environment details:
```sh
$ poetry env info

Virtualenv
Python:         3.12.3
Implementation: CPython
Path:           /home/kohki/.cache/pypoetry/virtualenvs/centralmanager-xA32UX7A-py3.12
Executable:     /home/kohki/.cache/pypoetry/virtualenvs/centralmanager-xA32UX7A-py3.12/bin/python
Valid:          True

Base
Platform:   linux
OS:         posix
Python:     3.12.3
Path:       /usr
Executable: /usr/bin/python3.12
```
If the Path and Executable fields under Virtualenv are NA, the virtual environment is not functioning properly.

Fix the virtual environment by specifying the correct Python version:
```sh
$ poetry env use python3.12
```
Then, verify again that the virtual environment is correctly configured and using the expected Python version.

Next, we will install the necessary libraries.  
Please run the following command.
```sh
$ poetry install
```

## How to use
To start the system, follow these steps:

Ensure that your camera and switch are connected to this PC.  

まず、config.tomlを書いてください。
```
[serial]
port = "/dev/ttyACM0"
baudrate = 9600

[directory]
image_dir = 'image/recorded'
dataset_dir = 'dataset'    
model_dir = 'model'

[model]
classes = ["0000", "0001", "0010", "0011"]
name = "mobilenetv3_small_latest.pth"
# name = "vgg_best_model.pth"
arch='mobilenet'
# arch='vgg'

[monitoring]
THRESHOLD = 10000  # File count threshold
CHECK_INTERVAL = 60

[hyperparameters]
# num_classes = 4 
batch_size = 32
epochs = 5
img_size = 224
learning_rate = 0.001

[gpu]
gpu_index = 1
```
[serial]の箇所では[serial]

Run the following command:
```sh
cd centralManager/
poetry run python main.py
```
To stop the system, enter `q` or `quit` in the terminal.

The log directory contains log files.
If any issues occur, please check the logs for details.

The image directory stores captured images.
Images are taken only when the system is in train mode, and they are captured 15 to 180 seconds after a switch input change.