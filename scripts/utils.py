import platform
import subprocess
from pathlib import Path
import re
import os

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
