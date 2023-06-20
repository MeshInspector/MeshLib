import os.path
import argparse
from subprocess import check_output, CalledProcessError

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="Output filename", default=None)
args = parser.parse_args()

VCPKG_CMD = "vcpkg export "
REQUIREMTS_PATH = "./requirements/windows.txt"

if not os.path.exists(REQUIREMTS_PATH):
    print("Cant find {}. Please, run this script from the repo root".format(REQUIREMTS_PATH))
    exit(1)

with open(REQUIREMTS_PATH) as fin:
    for line in fin.readlines():
        line = line.strip()
        if line:
            VCPKG_CMD += line + ":x64-windows-meshlib "

VCPKG_CMD += " --nuget"

# If the output filename is specified
if args.output:
    VCPKG_CMD += " --output " + args.output

try:
    check_output(VCPKG_CMD, shell=True)
except CalledProcessError as e:
    print(e.output)
    exit(1)
