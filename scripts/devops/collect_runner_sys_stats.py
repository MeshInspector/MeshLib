import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import subprocess

def get_ram_amount():
    system = platform.system()
    if system == "Darwin":
        output = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)
        return int(output)
    elif system == "Linux":
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    elif system == "Windows":
        ps_command = "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
        output = subprocess.check_output(['powershell', '-Command', ps_command], text=True)
        return int(output.strip())
    else:
        raise RuntimeError(f"Unknown system: {system}")

# decoding as in GetCpuId (source/MRMesh/MRSystem.cpp), but names must match
# lscpu (util-linux sys-utils/lscpu-arm.c) so old and new stats rows agree
ARM_CPU_NAMES = {
    (0x41, 0xd03): "Cortex-A53",     (0x41, 0xd05): "Cortex-A55",
    (0x41, 0xd07): "Cortex-A57",     (0x41, 0xd08): "Cortex-A72",
    (0x41, 0xd09): "Cortex-A73",     (0x41, 0xd0a): "Cortex-A75",
    (0x41, 0xd0b): "Cortex-A76",     (0x41, 0xd0c): "Neoverse-N1",
    (0x41, 0xd0d): "Cortex-A77",     (0x41, 0xd40): "Neoverse-V1",
    (0x41, 0xd41): "Cortex-A78",     (0x41, 0xd44): "Cortex-X1",
    (0x41, 0xd49): "Neoverse-N2",    (0x41, 0xd4f): "Neoverse-V2",
    (0xc0, 0xac3): "Ampere-1",       (0xc0, 0xac4): "Ampere-1a",
    (0x43, 0x0af): "ThunderX2-99xx", (0x46, 0x001): "A64FX",
    (0x51, 0xc01): "Saphira",
}

ARM_VENDORS = {
    0x41: "ARM",     0x42: "Broadcom",
    0x43: "Cavium",  0x48: "HiSilicon",
    0x4e: "NVIDIA",  0x51: "Qualcomm",
    0x53: "Samsung", 0x56: "Marvell",
    0x70: "Phytium", 0xc0: "Ampere",
}

def get_arm_cpu_model():
    implementer, part = -1, -1
    with open('/proc/cpuinfo') as f:
        for line in f:
            if line.startswith('CPU implementer'):
                implementer = int(line.split(':', 1)[1], 0)
            elif line.startswith('CPU part'):
                part = int(line.split(':', 1)[1], 0)
            if implementer >= 0 and part >= 0:
                break

    name = ARM_CPU_NAMES.get((implementer, part))
    if name:
        return name

    vendor = ARM_VENDORS.get(implementer)
    if vendor and part >= 0:
        return f"{vendor} ARM CPU (part {part:#x})"

    if implementer != -1 or part != -1:
        return f"ARM CPU: {implementer:#x}, {part:#x}"

    # single-board computers (Raspberry Pi etc.) expose a device-tree model name
    try:
        with open('/proc/device-tree/model') as f:
            name = f.read().strip('\0').strip()
            if name:
                return name
    except OSError:
        pass

    return "ARM CPU"

def get_cpu_model():
    system = platform.system()
    if system == "Darwin":
        output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True)
        return output.strip()
    elif system == "Linux":
        output = subprocess.check_output(['lscpu'], text=True)
        for line in output.splitlines():
            if line.startswith('Model name:'):
                model = line.split(':', 1)[1].strip()
                if model and model != '-':
                    return model
        if platform.machine() in ('aarch64', 'arm64'):
            return get_arm_cpu_model()
        return None
    elif system == "Windows":
        ps_command = "(Get-CimInstance Win32_Processor).Name"
        output = subprocess.check_output(['powershell', '-Command', ps_command], text=True)
        return output.strip().splitlines()[0]
    else:
        raise RuntimeError(f"Unknown system: {system}")

def get_free_disk_space():
    return shutil.disk_usage(os.environ.get('GITHUB_WORKSPACE', os.getcwd())).free

def get_compiler_id(compiler_path):
    # work-around for Windows runners
    if compiler_path.startswith("msvc-"):
        return compiler_path
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

        cpu_model = get_cpu_model()
        free_disk = math.floor(get_free_disk_space() / 1024 / 1024)

        results = {
            'target_os': os.environ.get('TARGET_OS'),
            'target_arch': os.environ.get('TARGET_ARCH'),
            'compiler': compiler_id,
            'build_config': os.environ.get('BUILD_CONFIG').lower(),
            'cpu_count': cpu_count,
            'cpu_model': cpu_model,
            'ram_mb': ram_amount,
            'free_disk_mb': free_disk,
            'build_system': build_system,
            'aws_instance_type': aws_instance_type or None,
        }
        with open(os.environ['STATS_FILE'], 'w') as f:
            json.dump(results, f)
            print(json.dumps(results, indent=2))
