import sys
import os
import shutil
from pathlib import Path
import fake_whl_helper as FWH

working_dir = Path(".").resolve()

def patch_targets_file( dir ):
    tfile = open( working_dir/ dir / "build" / "MeshLib.targets","w")
    tfile.write("<Project xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n")
    tfile.write("\t<ItemGroup>\n")
    for root,dir,files in os.walk(working_dir / dir / "content"):
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

def apply_patch(patch_dir,nuget_dir,clean):
    # remove old dependencies folder
    if clean:
        shutil.rmtree(working_dir / nuget_dir /"content")
    # copy mangled dependencies to proper location
    shutil.copytree(patch_dir + "/",nuget_dir+"/content/")
    # just clean directory
    shutil.rmtree(patch_dir)

def archive_nuget( dir, name ):
    shutil.make_archive(name,"zip",dir)
    os.rename(name+".zip",name)
    # clean working directory
    shutil.rmtree(dir)

input_nuget = sys.argv[1]

NUGET_DIR = "temp_nuget"
PATCH_DIR = "patch_dir"

extract_nuget(NUGET_DIR,input_nuget)
# create fake wheel to use common python repair tool that mangle wheel dependencies names (what we actually want to use for nuget package too)
FWH.make_fake_whl(working_dir / NUGET_DIR / "content" / "MRMeshC.dll")
FWH.patch_whl(PATCH_DIR,working_dir / NUGET_DIR / "content")

apply_patch(PATCH_DIR,NUGET_DIR,True) # first - windows patch: clean
for i in range(2,len(sys.argv)):
    apply_patch(sys.argv[i],NUGET_DIR,False) # all other patches without cleaning
    

patch_targets_file(NUGET_DIR)
# create new nuget archive in place of old one
archive_nuget(NUGET_DIR,input_nuget)
