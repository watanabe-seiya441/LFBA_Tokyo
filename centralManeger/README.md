# Poetry Environment Setup Guide
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