import platform
from pathlib import Path

MODULES = [
    "mrmeshpy",
    "mrmeshnumpy",
    "mrviewerpy",
]

WHEEL_SCRIPT_DIR = Path(__file__).parent.resolve()
WHEEL_ROOT_DIR = WHEEL_SCRIPT_DIR / "meshlib"
WHEEL_SRC_DIR = WHEEL_ROOT_DIR / "meshlib"
SOURCE_DIR = (WHEEL_SCRIPT_DIR / ".." / "..").resolve()

SYSTEM = platform.system()
LIB_EXTENSION = {
    'Darwin': ".so",
    'Linux': ".so",
    'Windows': ".pyd",
}[SYSTEM]
LIB_DIR = {
    'Darwin': SOURCE_DIR / "build" / "Release" / "bin" / "meshlib",
    'Linux': SOURCE_DIR / "build" / "Release" / "bin" / "meshlib",
    'Windows': SOURCE_DIR / "source" / "x64" / "Release",
}[SYSTEM]
