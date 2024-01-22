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
