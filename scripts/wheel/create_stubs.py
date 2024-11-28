import os
import shutil
import subprocess
import sys

from pathlib import Path


MODULES = [
    "mrmeshpy",
    "mrmeshnumpy",
    "mrviewerpy",
]

WHEEL_SCRIPT_DIR = Path(__file__).parent.resolve()
STUBS_ROOT_DIR = WHEEL_SCRIPT_DIR / "meshlib"
STUBS_SRC_DIR = STUBS_ROOT_DIR / "meshlib"
SOURCE_DIR = (WHEEL_SCRIPT_DIR / ".." / "..").resolve()

LIB_DIR = SOURCE_DIR / "build" / "Release" / "bin" / "meshlib"


def install_packages():
    packages = [
        "build",
        "pybind11-stubgen",
        "setuptools",
        "typing-extensions"
    ]

    subprocess.check_call(
        ["pip", "install", "--upgrade", "pip"]
    )
    subprocess.check_call(
        ["pip", "install", "--upgrade", *packages]
    )


def setup_workspace():
    if STUBS_ROOT_DIR.exists():
        shutil.rmtree(STUBS_ROOT_DIR)

    STUBS_SRC_DIR.mkdir(parents=True)

    init_file = LIB_DIR / "__init__.py"
    if init_file.exists():
        shutil.copy(init_file, STUBS_SRC_DIR / "__init__.py")
    else:
        shutil.copy(WHEEL_SCRIPT_DIR / "init.py", STUBS_SRC_DIR / "__init__.py")

    for module in MODULES:
        lib = LIB_DIR / f"{module}.so"
        shutil.copy(lib, STUBS_SRC_DIR)

def generate_stubs():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(STUBS_ROOT_DIR)

    os.chdir(STUBS_ROOT_DIR)
    for module in MODULES:
        subprocess.check_call(
            ["pybind11-stubgen", "--exit-code", "--output-dir", ".", f"meshlib.{module}"],
            env=env,
        )

if __name__ == "__main__":
    try:
        install_packages()
        setup_workspace()
        generate_stubs()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
