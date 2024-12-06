import math
import multiprocessing
import os
import platform
import re
import subprocess

def get_ram_amount():
    system = platform.system()
    if system == "Darwin":
        output = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)
        return int(output)
    elif system == "Linux":
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    elif system == "Windows":
        output = subprocess.check_output(['wmic', 'ComputerSystem', 'get', 'TotalPhysicalMemory'], text=True)
        return int(re.search(r'\d+', output).group())
    else:
        raise RuntimeError(f"Unknown system: {system}")

def get_compiler_id(compiler_path):
    output = subprocess.check_output([compiler_path, "--version"]).decode()
    version_line = output.splitlines()[0]
    if version_line.startswith("Apple clang"):
        compiler = "appleclang"
    elif "clang" in version_line:
        compiler = "clang"
    elif "g++" in version_line or "gcc" in version_line:
        compiler = "gcc"
    else:
        raise RuntimeError(f"Unknown compiler: {version_line}")

    output = subprocess.check_output([compiler_path, "-dumpversion"]).decode()
    major_version = re.match(r"(\d+)", output).group(1)

    return f"{compiler}-{major_version}"

if __name__ == "__main__":
    print(f"CPU_COUNT={multiprocessing.cpu_count()}")
    print(f"RAM_AMOUNT={math.floor(get_ram_amount() / 1024 / 1024)}")
    if cxx_compiler := os.environ.get("CXX_COMPILER"):
        print(f"COMPILER_ID={get_compiler_id(cxx_compiler)}")
