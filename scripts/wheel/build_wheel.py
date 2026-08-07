import functools
import os
import platform
import shutil
import subprocess
import sys
import re

from argparse import ArgumentParser
from string import Template

from build_constants import *
import create_stubs
import split_wheel

def install_packages():
    create_stubs.install_packages()

    packages = [
        "build",
        "setuptools",
        "wheel",
        "numpy", # Because the modules we're building depend on it.
    ]

    platform_specific_packages = {
        'Darwin': [
            "delocate==0.10.7",
        ],
        'Linux': [
            "auditwheel",
        ],
        'Windows': [
            "delvewheel",
        ],
    }
    packages += platform_specific_packages[SYSTEM]

    subprocess.check_call(
        ["pip", "install", "--upgrade", "pip"]
    )
    subprocess.check_call(
        ["pip", "install", "--upgrade", *packages]
    )


def setup_workspace(version, modules, plat_name):
    if WHEEL_ROOT_DIR.exists():
        shutil.rmtree(WHEEL_ROOT_DIR)

    WHEEL_SRC_DIR.mkdir(parents=True)

    create_stubs.setup_workspace(modules, False)

    print("Copying LICENSE and readme.md")
    shutil.copy(SOURCE_DIR / "LICENSE", WHEEL_ROOT_DIR)
    shutil.copy(SOURCE_DIR / "readme.md", WHEEL_ROOT_DIR)

    shutil.copy(SOURCE_DIR / "thirdparty" / "licenses" / "THIRD-PARTY-NOTICES.txt", WHEEL_ROOT_DIR)

    print("Copying resource files...")
    shutil.copy(SOURCE_DIR / "source" / "MRViewer" / "MRDarkTheme.json", WHEEL_SRC_DIR)
    shutil.copy(SOURCE_DIR / "source" / "MRViewer" / "MRLightTheme.json", WHEEL_SRC_DIR)
    shutil.copy(SOURCE_DIR / "thirdparty" / "fontawesome-free" / "fa-solid-900.ttf", WHEEL_SRC_DIR)
    shutil.copytree(SOURCE_DIR / "thirdparty" / "Noto_Sans", WHEEL_SRC_DIR, dirs_exist_ok=True)
    shutil.copytree(SOURCE_DIR / "source" / "MRViewer" / "resource", WHEEL_SRC_DIR / "resource", dirs_exist_ok=True )
    icon_resources = [
        str(icon_resource.relative_to(WHEEL_SRC_DIR))
        for icon_resource in (WHEEL_SRC_DIR / "resource").rglob("*.*") # no folders
    ]
    font_resources = [
        str(font_resources.relative_to(WHEEL_SRC_DIR))
        for font_resources in (WHEEL_SRC_DIR).glob("NotoSans*.*") # no folders
    ]
    pybind_shims = []
    py_versions = []
    for pybind_shim in LIB_DIR_MESHLIB.glob("*pybind11nonlimitedapi_meshlib_*"):
        shutil.copy(pybind_shim, WHEEL_SRC_DIR)
        pybind_shim_name = os.path.basename(pybind_shim)
        pybind_shims.append(pybind_shim_name)
        py_versions.append(int(re.sub("\\..*", "", re.sub(".*pybind11nonlimitedapi_meshlib_3\\.", "", pybind_shim_name))));
    py_versions.sort()

    shutil.copy(WHEEL_SCRIPT_DIR / "pyproject.toml", WHEEL_ROOT_DIR)

    # generate setup.cfg
    package_files = [
        *pybind_shims,
        *icon_resources,
        "MRDarkTheme.json",
        "MRLightTheme.json",
        "fa-solid-900.ttf",
        *font_resources
    ]
    for module in modules:
        package_files += [
            f"{module}{LIB_EXTENSION}",
            f"{module}.pyi",
        ]
    with open(WHEEL_SCRIPT_DIR / "setup.cfg.in", 'r') as config_template_file:
        config = Template(config_template_file.read()).substitute(
            VERSION=version,
            PACKAGE_DATA=", ".join(package_files),
            PYTHON_TAG=".".join(f"py3{x}" for x in py_versions),
            PLAT_NAME=plat_name,
        )
    with open(WHEEL_ROOT_DIR / "setup.cfg", 'w') as config_file:
        config_file.write(config)


def strip_libraries():
    # Only MeshLib's own libs need this: the vcpkg-built third-party libs are already stripped.
    # Must run before `auditwheel repair`, not after (e.g. via its --strip flag): auditwheel
    # patchelf's the grafted libs, and stripping a patchelf'ed lib breaks its load command
    # alignment, making it unloadable.
    if SYSTEM != "Linux":
        return
    libs = [
        *LIB_DIR.glob("libMR*.so"),
        *LIB_DIR.glob("libpybind11nonlimitedapi_stubs.so"),
        *WHEEL_SRC_DIR.glob("*.so"),
    ]
    for lib in libs:
        subprocess.check_call(["strip", "--strip-all", lib])


def build_wheel():
    os.chdir(WHEEL_ROOT_DIR)
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel"]
    )

    full_wheel_file = list(WHEEL_ROOT_DIR.glob("dist/*.whl"))[0]
    # The viewer-less copy is repaired alongside the full wheel, so the repair tool itself
    # decides which bundled libraries are viewer-only (see split_wheel.py).
    core_wheel_file = split_wheel.make_core_input(full_wheel_file, WHEEL_ROOT_DIR / "dist_core")

    if SYSTEM == "Linux":
        # see also: https://github.com/mayeut/pep600_compliance
        manylinux_version = "2_28"

        os.chdir(WHEEL_ROOT_DIR)
        # the repaired core wheel ships as `meshlib-core`; the repaired full wheel is
        # temporary - only its complement ships, as the `meshlib` wheel
        for wf, out_dir in ((full_wheel_file, "wheelhouse_full"), (core_wheel_file, "wheelhouse")):
            subprocess.check_call(
                [
                    sys.executable, "-m", "auditwheel",
                    "repair",
                    "--plat", f"manylinux_{manylinux_version}_{platform.machine()}",
                    "-w", out_dir,
                    wf
                ]
            )

        split_wheel.extract_meshlib_wheel(
            next((WHEEL_ROOT_DIR / "wheelhouse_full").glob("meshlib_core-*.whl")),
            next((WHEEL_ROOT_DIR / "wheelhouse").glob("meshlib_core-*.whl")),
        )

    elif SYSTEM == "Windows":
        os.chdir(SOURCE_DIR)
        # the repaired core wheel ships as `meshlib-core`; the repaired full wheel is
        # temporary - only its complement ships, as the `meshlib` wheel
        for wf, out_dir in ((full_wheel_file, "wheelhouse_full"), (core_wheel_file, "wheelhouse")):
            subprocess.check_call(
                [
                    sys.executable, "-m", "delvewheel",
                    "repair",
                    # We use --no-dll "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll" here to avoid strange conflict
                    # that happens if we pack these dlls into whl.
                    # Another option is to use --no-mangle "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll"
                    # to pack these dlls with original names and let system solve conflicts on import
                    # https://stackoverflow.com/questions/78817088/vsruntime-dlls-conflict-after-delvewheel-repair
                    # UPDATE:
                    #  no longer needed due to https://github.com/adang1345/delvewheel/issues/49 fix with https://github.com/adang1345/delvewheel/commit/42a52cdcc15d424b030a94cb4b51a6b72e4a3d92
                    #"--no-dll", "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll",
                    "--add-path", LIB_DIR,
                    # This is needed to catch our `pybind11nonlimitedapi_meshlib_3.X.dll` on Windows. Otherwise they don't get patched,
                    # and then can't find `pybind11nonlimitedapi_stubs.dll`, which does get patched.
                    "--analyze-existing",
                    "-w", out_dir,
                    wf
                ]
            )
        split_wheel.extract_meshlib_wheel(
            next((SOURCE_DIR / "wheelhouse_full").glob("meshlib_core-*.whl")),
            next((SOURCE_DIR / "wheelhouse").glob("meshlib_core-*.whl")),
        )

    elif SYSTEM == "Darwin":
        os.chdir(WHEEL_ROOT_DIR)
        subprocess.check_call(
            ["delocate-path", "meshlib"]
        )
        os.chdir(SOURCE_DIR)
        # the repaired core wheel ships as `meshlib-core`; the repaired full wheel is
        # temporary - only its complement ships, as the `meshlib` wheel
        for wf, out_dir in ((full_wheel_file, "./wheelhouse_full"), (core_wheel_file, ".")):
            subprocess.check_call(
                ["delocate-wheel", "-w", out_dir, "-v", wf]
            )
        split_wheel.extract_meshlib_wheel(
            next((SOURCE_DIR / "wheelhouse_full").glob("meshlib_core-*.whl")),
            next(SOURCE_DIR.glob("meshlib_core-*.whl")),
        )


if __name__ == "__main__":
    csv = functools.partial(str.split, sep=",")

    parser = ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--modules", type=csv, default=MODULES)
    parser.add_argument("--plat-name", default="any")
    args = parser.parse_args()

    try:
        install_packages()
        setup_workspace(version=args.version, modules=args.modules, plat_name=args.plat_name)
        create_stubs.generate_stubs(modules=args.modules)
        strip_libraries()
        build_wheel()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
