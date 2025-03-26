import sys
import os
import shutil
import subprocess

input_nuget = sys.argv[1]
shutil.unpack_archive(input_nuget,"temp_nuget","zip")

# create fake wheel to use common python repair tool that mangle wheel dependencies names (what we actually want to use for nuget package too)
os.mkdir("dummy_wheel",)
os.mkdir("dummy_wheel/dummy.libs")
os.mkdir("dummy_wheel/dummy-1.0.dist-info")
# add only used dll in fake wheel
shutil.copyfile("temp_nuget/content/MRMeshC.dll","dummy_wheel/dummy.libs/MRMeshC.dll")
# actually create whl file
shutil.make_archive("dummy-1.0-py3-none-any","zip","dummy_wheel")
os.rename("dummy-1.0-py3-none-any.zip","dummy-1.0-py3-none-any.whl")

# use mangling tool on whl file
# store result dlls in `content/dummy.libs/`
try:
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

            "--add-path", "temp_nuget/content", # path where input dependencies are located

            # use this directory instead of extracting files from result whl file
            "--extract-dir", "content",

            # main option - needed to mangle whl/libs/ content (only thing we doing it for)
            "--analyze-existing",
            "dummy-1.0-py3-none-any.whl"
        ]
    )
except subprocess.CalledProcessError as e:
    print(e)
    sys.exit(e.returncode)

# remove old dependencies folder
shutil.rmtree("temp_nuget/content")
# copy mangled dependencies to proper location
shutil.copytree("content/dummy.libs/","temp_nuget/content/")
# just clean directory
shutil.rmtree("dummy_wheel")
shutil.rmtree("wheelhouse")
shutil.rmtree("content")
os.remove("dummy-1.0-py3-none-any.whl")
# remove input file to replace it with new one
os.remove(input_nuget)
# create new nuget archive in place of old one
shutil.make_archive(input_nuget,"zip","temp_nuget")
os.rename(input_nuget+".zip",input_nuget)
# clean working directory
shutil.rmtree("temp_nuget")
