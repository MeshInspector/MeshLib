import platform
from pathlib import Path

def get_build_consts():
    res = {}
    res['MODULES'] = [
        "mrmeshpy",
        "mrmeshnumpy",
        "mrviewerpy",
    ]

    res['WHEEL_SCRIPT_DIR'] = Path(__file__).parent.resolve()
    res['WHEEL_ROOT_DIR'] = res['WHEEL_SCRIPT_DIR'] / "meshlib"
    res['WHEEL_SRC_DIR'] = res['WHEEL_ROOT_DIR'] / "meshlib"
    res['SOURCE_DIR'] = (res['WHEEL_SCRIPT_DIR'] / ".." / "..").resolve()

    res['SYSTEM'] = platform.system()
    res['LIB_EXTENSION'] = {
        'Darwin': ".so",
        'Linux': ".so",
        'Windows': ".pyd",
    }[res['SYSTEM']]
    res['LIB_DIR'] = {
        'Darwin': res['SOURCE_DIR'] / "build" / "Release" / "bin" / "meshlib",
        'Linux': res['SOURCE_DIR'] / "build" / "Release" / "bin" / "meshlib",
        'Windows': res['SOURCE_DIR'] / "source" / "x64" / "Release",
    }[res['SYSTEM']]

    return res