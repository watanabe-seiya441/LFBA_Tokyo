
------メモ--------

## Poetryの環境構築の仕方
公式インストーラーを使用
```
$ curl -sSL https://install.python-poetry.org | python3 -
```
versionを確認
```
$ poetry --version
Poetry (version 2.1.1)
```
また、仮想環境を確認する。
```
$ poetry run python -c "import sys; print(sys.executable)"
/home/kohki/.cache/pypoetry/virtualenvs/centralmaneger-xA32UX7A-py3.12/bin/python
```
もし、これがホストpythonの場合、仮想環境が上手く動いていない可能性がある。
```
$ poetry env info

Virtualenv
Python:         3.12.3
Implementation: CPython
Path:           /home/kohki/.cache/pypoetry/virtualenvs/centralmaneger-xA32UX7A-py3.12
Executable:     /home/kohki/.cache/pypoetry/virtualenvs/centralmaneger-xA32UX7A-py3.12/bin/python
Valid:          True

Base
Platform:   linux
OS:         posix
Python:     3.12.3
Path:       /usr
Executable: /usr/bin/python3.12
```
ここで、VirtualenvのPathとExecutableがNAの場合、上手く動いていない。
この時、以下を実行する。
```
$ poetry env use python3.12
```
もう一度、仮想環境を確認して、しっかり仮想環境のPythonで動いていることを確認する。