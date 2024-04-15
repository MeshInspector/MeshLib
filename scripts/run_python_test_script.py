import os
import sys
import platform
import argparse

parser = argparse.ArgumentParser(description="Python Test Script")

parser.add_argument("-cmd", dest="cmd", type=str, help='Overwrite python run cmd')
parser.add_argument("-d", dest="dir", type=str, help='Path to tests')
parser.add_argument("-m", dest="marks", type=str, help='Test marks (tags) to apply into pytest')

args = parser.parse_args()
print(args)

python_cmd = "py -3.11 "
platformSystem = platform.system()

if platformSystem == 'Linux':
    python_cmd = "python3 "

    os_name = ""
    os_version = ""
    if os.path.exists('/etc/os-release'):
        lines = open('/etc/os-release').read().split('\n')
        for line in lines:
            if line.startswith('NAME='):
                os_name = line.split('=')[-1].replace('"', '')
            if line.startswith('VERSION_ID='):
                os_version = line.split('=')[-1].replace('"', '')

    if "ubuntu" in os_name.lower():
        if os_version.startswith("20"):
            python_cmd = "python3.8 "
        elif os_version.startswith("22"):
            python_cmd = "python3.10 "
    elif "fedora" in os_name.lower():
        if os_version.startswith("35"):
            python_cmd = "python3.9 "
        elif os_version.startswith("37"):
            python_cmd = "python3.11 "
        elif os_version.startswith("39"):
            python_cmd = "python3.12 "

elif platformSystem == 'Darwin':
    python_cmd = "python3.10 "

if args.cmd:
    python_cmd = str(args.cmd).strip() + " "

directory = os.getcwd()
try:
    directory = os.path.dirname(os.path.abspath(__file__))
except NameError:  # embedded python exception
    print("trying to resolve path manually...")
    directory = os.path.join(directory, "../../../MeshLib/")
    directory = os.path.join(directory, "test_python")
    print(directory)

if args.dir:
    directory = os.path.join(directory, args.dir)
else:
    directory = os.path.join(directory, "..")
    directory = os.path.join(directory, "test_python")

os.environ["MeshLibPyModulesPath"] = os.getcwd()
os.chdir(directory)

os.system(python_cmd + "-m pip uninstall -y meshlib")
pytest_cmd = "-m pytest -s -v --basetemp=../pytest_temp --durations 30"
if args.marks:
    pytest_cmd += f' -m "{args.marks}"'
res = os.system(python_cmd + pytest_cmd)

if res != 0:
    sys.exit(1)
