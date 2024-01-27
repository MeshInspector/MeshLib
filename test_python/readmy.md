## Contribution Instructions


1. Install the required linting packages: `pip3 install pre-commit black isort`
2. Make sure to run `pre-commit install` to install the commit hook to automatically check for linting on commiting.
3. When commiting the black & isort check should automatically reformat any edited python code.

To run this yourself you can do:
- `black ./test_python`
- `isort ./test_python/ --profile black`

As an extra check there is also a workflow (`python-lint.yml`) that automatically checks if the linting is satisfied on creating a PR.


## Setup Instructions

To run these tests please build MeshLib first, then do following commands
on linux or mac:
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
