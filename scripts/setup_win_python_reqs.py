import platform
import subprocess
from pathlib import Path

def get_vcpkg_root_from_where():
    try:
        output = subprocess.check_output(["where", "vcpkg"], universal_newlines=True)
        first_line = output.strip().splitlines()[0]
        return Path(first_line).resolve().parent
    except Exception as e:
        print(e)
        return None

def detect_vcpkg_python_version(vcpkg_root, triplet="x64-windows-meshlib"):
    include_dir = vcpkg_root / "installed" / triplet / "include"
    if include_dir.exists():
        for entry in include_dir.iterdir():
            match = re.match(r"python3\.(\d+)", entry.name)
            if match:
                minor_version = match.group(1)
                return f"3.{minor_version}"
    return None

def choco_install_python(version: str):
    version_nodot = version.replace('.', '')  # "3.11" -> "311"
    choco_package = f"python{version_nodot}"
    try:
        print(f"Installing {choco_package} via Chocolatey...")
        subprocess.run(["choco", "install", choco_package, "-y"], check=True)
        print(f"Successfully installed {choco_package}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {choco_package}. Check Chocolatey setup and try again.")

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
