import os
import shutil
import platform
import subprocess
import sys
from pathlib import Path

SYSTEM = platform.system()

def make_fake_whl(dll_path : Path):
    w_dir = Path(".").resolve()
    whl_dir = w_dir / "temp_whl_dir"
    os.mkdir(whl_dir)
    whl_libs_path = whl_dir / "dummy.libs"
    whl_info_path = whl_dir / "dummy-1.0.dist-info"
    os.mkdir( whl_libs_path)
    os.mkdir( whl_info_path )
    # copy dll
    shutil.copyfile(dll_path, whl_libs_path / dll_path.name )
    # create servant files
    with open( whl_info_path / "WHEEL", "w" ) as wheel_file:
        pass
    with open( whl_info_path / "RECORD", "w" ) as record_file:
        record_file.write( "dummy.libs/" + dll_path.name )
    # actually create whl file
    shutil.make_archive("dummy-1.0-py3-none-any","zip",whl_dir)
    os.rename("dummy-1.0-py3-none-any.zip","dummy-1.0-py3-none-any.whl")
    # clean
    shutil.rmtree(whl_dir)

def patch_whl(out_dir,libs_dir):
    # use mangling tool on whl file
    # store result dlls in `content/dummy.libs/`
    try:
        if SYSTEM == "Windows":
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

                    "--add-path",libs_dir, # path where input dependencies are located

                    # main option - needed to mangle whl/libs/ content (only thing we doing it for)
                    "--analyze-existing",
                    "dummy-1.0-py3-none-any.whl"
                ]
            )
        elif SYSTEM == "Linux":
            sys.path.append(libs_dir) # to find SO files
            # see also: https://github.com/mayeut/pep600_compliance
            manylinux_version = "2_31"
            subprocess.check_call(
                [
                    sys.executable, "-m", "auditwheel",
                    "repair",
                    "--plat", f"manylinux_{manylinux_version}_{platform.machine()}",
                    "dummy-1.0-py3-none-any.whl"
                ]
            )
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(e.returncode)
    # not needed anymore
    os.remove("dummy-1.0-py3-none-any.whl")

    repaired_files = []
    for repaired_wheel_file in Path(".").glob("wheelhouse/dummy-*.whl"):
        repaired_files.append(repaired_wheel_file)
    shutil.unpack_archive(repaired_files[0],"patched_whl","zip")
    shutil.copytree(Path("patched_whl") / "dummy.libs/",Path(out_dir))
    #clean
    shutil.rmtree("wheelhouse")
    shutil.rmtree("patched_whl")
