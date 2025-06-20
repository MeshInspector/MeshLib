import json
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
    # work-around for Windows runners
    if compiler_path.startswith("windows-"):
        return compiler_path.replace("windows-", "msvc-")

    output = subprocess.check_output([compiler_path, "--version"]).decode()
    version_line = output.splitlines()[0]
    if version_line.startswith("Apple clang"):
        compiler = "appleclang"
    elif version_line.startswith("emcc"):
        compiler = "emcc"
    elif "clang" in version_line:
        compiler = "clang"
    elif "g++" in version_line or "gcc" in version_line:
        compiler = "gcc"
    else:
        raise RuntimeError(f"Unknown compiler: {version_line}")

    output = subprocess.check_output([compiler_path, "-dumpversion"]).decode().strip()
    if compiler == "emcc":
        return f"emcc-{output}"
    else:
        major_version = re.match(r"(\d+)", output).group(1)
        return f"{compiler}-{major_version}"

if __name__ == "__main__":
    if os.environ.get('CI'):
        cpu_count = multiprocessing.cpu_count()
        ram_amount = math.floor(get_ram_amount() / 1024 / 1024)

        with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
            print(f"cpu_count={cpu_count}", file=out)
            print(f"ram_amount_mb={ram_amount}", file=out)

        compiler_id = None
        cxx_compiler = os.environ.get("CXX_COMPILER")
        if cxx_compiler:
            compiler_id = get_compiler_id(cxx_compiler)

        build_system = os.environ.get('BUILD_SYSTEM').lower()
        if not build_system:
            build_system = "msbuild" if compiler_id.startswith("msvc") else "cmake"

        aws_instance_type = os.environ.get('AWS_INSTANCE_TYPE', '').lower()

        results = {
            'target_os': os.environ.get('TARGET_OS'),
            'target_arch': os.environ.get('TARGET_ARCH'),
            'compiler': compiler_id,
            'build_config': os.environ.get('BUILD_CONFIG').lower(),
            'cpu_count': cpu_count,
            'ram_mb': ram_amount,
            'build_system': build_system,
            'aws_instance_type': aws_instance_type or None,
        }
        with open(os.environ['STATS_FILE'], 'w') as f:
            json.dump(results, f)
            print(json.dumps(results, indent=2))
