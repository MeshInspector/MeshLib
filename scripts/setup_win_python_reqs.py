import platform
import subprocess

# Add current script directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")))

from utils import get_vcpkg_root_from_where, detect_vcpkg_python_version, choco_install_python
from pathlib import Path

def run_python_setup(py_version: str):
    py_cmd = f"py -{py_version}"
    try:
        print(f"Ensuring pip is available for Python {py_version}...")
        subprocess.run(f"{py_cmd} -m ensurepip --upgrade", shell=True, check=True)
        subprocess.run(f"{py_cmd} -m pip install --upgrade pip", shell=True, check=True)

        requirements_file = Path(__file__).resolve().parents[1] / "requirements" / "python.txt"
        if requirements_file.exists():
            subprocess.run(f"{py_cmd} -m pip install -r {requirements_file}", shell=True, check=True)
        else:
            print(f"WARNING: Requirements file not found at {requirements_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during Python setup for {py_version}: {e}")
        exit(1)

if platform.system() == "Windows":
    vcpkg_root = get_vcpkg_root_from_where()
    if vcpkg_root:
        detected_version = detect_vcpkg_python_version(vcpkg_root)
        if detected_version:
            print(f"Using python version {detected_version}")
            choco_install_python(detected_version)
            run_python_setup(detected_version)
