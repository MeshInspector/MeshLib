import os.path
from subprocess import check_output

VCPKG_CMD = "vcpkg export "
REQUIREMTS_PATH = "./requirements/windows.txt"

if not os.path.exists(REQUIREMTS_PATH):
    print("Cant find {}. Please, run this script from the repo root".format(REQUIREMTS_PATH))

with open(REQUIREMTS_PATH) as fin:
    for line in fin.readlines():
        VCPKG_CMD += line.strip() + ":x64-windows-meshlib "

VCPKG_CMD += " --nuget"

check_output("VCPKG_CMD", shell=True)
