import sys
import os
import shutil
import subprocess

input_nuget = sys.argv[1]
shutil.unpack_archive(input_nuget,"temp_nuget","zip")
os.mkdir("dummy_wheel",)
os.mkdir("dummy_wheel/dummy.libs")
os.mkdir("dummy_wheel/dummy-1.0.dist-info")

shutil.copyfile("temp_nuget/content/MRMeshC.dll","dummy_wheel/dummy.libs/MRMeshC.dll")

shutil.make_archive("dummy-1.0-py3-none-any","zip","dummy_wheel")
os.rename("dummy-1.0-py3-none-any.zip","dummy-1.0-py3-none-any.whl")

subprocess.check_call(
    [
        sys.executable, "-m", "delvewheel",
        "repair",
        # We use --no-dll "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll" here to avoid strange conflict
        # that happens if we pack these dlls into whl.
        # Another option is to use --no-mangle "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll"
        # to pack these dlls with original names and let system solve conflicts on import
        # https://stackoverflow.com/questions/78817088/vsruntime-dlls-conflict-after-delvewheel-repair
        "--no-mangle", "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll",
        "--add-path", "temp_nuget/content",
        "--extract-dir", "content",

        # This is needed to catch our `pybind11nonlimitedapi_meshlib_3.X.dll` on Windows. Otherwise they don't get patched,
        # and then can't find `pybind11nonlimitedapi_stubs.dll`, which does get patched.
        "--analyze-existing",
        "dummy-1.0-py3-none-any.whl"
    ]
)

shutil.rmtree("temp_nuget/content")
shutil.copytree("content/dummy.libs/","temp_nuget/content/")
shutil.rmtree("dummy_wheel")
shutil.rmtree("wheelhouse")
shutil.rmtree("content")
os.remove("dummy-1.0-py3-none-any.whl")
os.remove(input_nuget)
shutil.make_archive(input_nuget,"zip","temp_nuget")
os.rename(input_nuget+".zip",input_nuget)
shutil.rmtree("temp_nuget")
