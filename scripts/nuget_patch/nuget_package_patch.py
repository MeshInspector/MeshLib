import sys
import os
import shutil
import subprocess
from pathlib import Path

def patch_targets_file( dir ):
    tfile = open( Path(dir) / "build" / "MeshLib.targets","w")
    tfile.write("<Project xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n")
    tfile.write("\t<ItemGroup>\n")
    for root,dir,files in os.walk(Path(dir) / "content"):
        for file in files:
            if file.endswith(".dll") or ".so" in file:
                tfile.write("\t\t<None Include=\"$(MSBuildThisFileDirectory)\\..\\content\\"+file+"\">\n")
                tfile.write("\t\t\t<Link>"+file+"</Link>\n")
                tfile.write("\t\t\t<Visible>false</Visible>\n")
                tfile.write("\t\t\t<CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>\n")
                tfile.write("\t\t</None>\n")

    tfile.write("\t</ItemGroup>\n")
    tfile.write("</Project>\n")

def extract_nuget( dir, name ):
    shutil.unpack_archive(name,dir,"zip")
    # remove input file to replace it with new one
    os.remove(name)

def make_fake_whl(whl_dir,nuget_dir):
    os.mkdir(whl_dir)
    whl_libs_path = Path(whl_dir) / "dummy.libs"
    os.mkdir( whl_libs_path)
    os.mkdir( Path(whl_dir) /"dummy-1.0.dist-info")
    # add only used dll in fake wheel
    dll_name = "MRMeshC.dll"
    dll_path = Path(nuget_dir) / "content" / dll_name
    shutil.copyfile(dll_path, whl_libs_path /dll_name)
    # actually create whl file
    shutil.make_archive("dummy-1.0-py3-none-any","zip",whl_dir)
    os.rename("dummy-1.0-py3-none-any.zip","dummy-1.0-py3-none-any.whl")
    # clean
    shutil.rmtree(whl_dir)

def patch_whl(out_dir,nuget_dir):
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

                "--add-path", Path(nuget_dir) / "content", # path where input dependencies are located

                # use this directory instead of extracting files from result whl file
                "--extract-dir", out_dir,

                # main option - needed to mangle whl/libs/ content (only thing we doing it for)
                "--analyze-existing",
                "dummy-1.0-py3-none-any.whl"
            ]
        )
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(e.returncode)
    # not needed anymore
    os.remove("dummy-1.0-py3-none-any.whl")

def apply_patch(patch_dir,nuget_dir):
    # remove old dependencies folder
    shutil.rmtree(Path(nuget_dir)/"content")
    # copy mangled dependencies to proper location
    shutil.copytree(patch_dir + "/dummy.libs/",nuget_dir+"/content/")
    # just clean directory
    shutil.rmtree("wheelhouse")
    shutil.rmtree(patch_dir)

def archive_nuget( dir, name ):
    shutil.make_archive(name,"zip",dir)
    os.rename(name+".zip",name)
    # clean working directory
    shutil.rmtree(dir)

input_nuget = sys.argv[1]

extract_nuget("temp_nuget",input_nuget)
# create fake wheel to use common python repair tool that mangle wheel dependencies names (what we actually want to use for nuget package too)
make_fake_whl("dummy_wheel","temp_nuget")
patch_whl("content","temp_nuget")
apply_patch("content","temp_nuget")
patch_targets_file("temp_nuget")
# create new nuget archive in place of old one
archive_nuget("temp_nuget",input_nuget)
