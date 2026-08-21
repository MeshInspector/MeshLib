## Contribution Instructions


1. Install the linting packages: `pip3 install pre-commit black isort`
3. When commiting the black & isort check should automatically reformat any edited python code.

To run this yourself you can do:
- `black ./test_python`
- `isort ./test_python/ --profile black`


## Setup Instructions

To run these tests please build MeshLib first, then do following commands
on linux:
```sh
python3 -m pip install pytest numpy
cd build/Debug/bin # or build/Release/bin
# `xvfb-run -a` gives the mrviewerpy tests a display; without it they skip themselves
xvfb-run -a python3 ./../../../scripts/run_python_test_script.py -d '../test_python'
```

on mac:
```sh
python3 -m pip install pytest numpy
cd build/Debug/bin # or build/Release/bin
python3 ./../../../scripts/run_python_test_script.py -d '../test_python'
```

on Windows:
```shell
py -3 -m pip install pytest numpy
cd source/x64/Debug # or source/x64/Release
py -3 ..\..\..\scripts\run_python_test_script.py -d '..\test_python'
```
