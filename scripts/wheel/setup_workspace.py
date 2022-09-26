import glob
import os
import platform
import shutil
import sys

WHEEL_SRC_DIR = os.path.join(os.getcwd(), "scripts/wheel/meshlib/meshlib/")
WHEEL_ROOT_DIR = os.path.join(os.getcwd(), "scripts/wheel/meshlib/")
WHEEL_SCRIPT_DIR = os.path.join(os.getcwd(), "scripts/wheel/")

PYLIB_PATH = {"Windows": r'./source/x64/Release/*.pyd',
              "Linux": r'./build/Release/bin/meshlib/mr*.so',
              "Darwin": r'./build/Release/bin/meshlib/mr*.so'}


def prepare_workspace():
    if not os.path.isdir(os.path.join(os.getcwd(), "scripts")):
        print("Please run this script from MeshLib root")
        sys.exit(1)

    if os.path.exists(WHEEL_ROOT_DIR):
        shutil.rmtree(WHEEL_ROOT_DIR)

    os.makedirs(WHEEL_SRC_DIR, exist_ok=True)
    print("Copying LICENSE and readme.md")
    shutil.copy("LICENSE", WHEEL_ROOT_DIR)
    shutil.copy("readme.md", WHEEL_ROOT_DIR)
    # create empty file
    open(os.path.join(WHEEL_SRC_DIR, "__init__.py"), "w").close()


def copy_src():
    platform_system = platform.system()
    print("Copying {} files...".format(platform_system))
    for file in glob.glob(PYLIB_PATH[platform_system]):
        print(file)
        shutil.copy(file, WHEEL_SRC_DIR)

    shutil.copy(os.path.join(WHEEL_SCRIPT_DIR, "setup.py"), WHEEL_ROOT_DIR)


prepare_workspace()
copy_src()
