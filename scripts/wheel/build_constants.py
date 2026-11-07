import platform
from pathlib import Path

SYSTEM = platform.system()

# Windows output dirs are named after the target architecture; the building
# interpreter is native, so its own machine tells us which one.
WIN_ARCH = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"

MODULES = [
    "mrmeshpy",
    "mrmeshnumpy",
    "mrviewerpy",
    "mrcudapy",
]


WHEEL_SCRIPT_DIR = Path(__file__).parent.resolve()
WHEEL_ROOT_DIR = WHEEL_SCRIPT_DIR / "meshlib"
WHEEL_SRC_DIR = WHEEL_ROOT_DIR / "meshlib"
SOURCE_DIR = (WHEEL_SCRIPT_DIR / ".." / "..").resolve()

LIB_EXTENSION = {
    'Darwin': ".so",
    'Linux': ".so",
    'Windows': ".pyd",
}[SYSTEM]
LIB_DIR = {
    'Darwin': SOURCE_DIR / "build" / "Release" / "bin",
    'Linux': SOURCE_DIR / "build" / "Release" / "bin",
    'Windows': SOURCE_DIR / "source" / WIN_ARCH / "Release",
}[SYSTEM]
LIB_DIR_MESHLIB = LIB_DIR / "meshlib"

# modules written in Python, not compiled; their stubs are written by hand, next to them
PYTHON_SOURCE_MODULES = [
    "mrmeshnumpy",
]


def module_file_name(module):
    # Without Cuda, `mrcudapy` is a placeholder written in Python too.
    is_python = module in PYTHON_SOURCE_MODULES or (LIB_DIR_MESHLIB / f"{module}.py").exists()
    return f"{module}.py" if is_python else f"{module}{LIB_EXTENSION}"
