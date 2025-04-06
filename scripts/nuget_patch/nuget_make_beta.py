import sys
import os
import shutil
from pathlib import Path

INPUT_FILE_NAME = sys.argv[1]
NUGET_TEMP_DIR = Path("nuget_temp_dir")

shutil.unpack_archive(INPUT_FILE_NAME,NUGET_TEMP_DIR,"zip")
# remove input file to replace it with new one
os.remove(INPUT_FILE_NAME)


SPEC_FILE = NUGET_TEMP_DIR / "MeshLib.nuspec"
OLD_SPEC_FILE = NUGET_TEMP_DIR / "MeshLib_old.nuspec"
os.rename(SPEC_FILE,OLD_SPEC_FILE)
with open(OLD_SPEC_FILE) as fin, open(SPEC_FILE, 'w') as fout:
    for line in fin:
        lineout = line.replace("</version>","-beta</version>")
        fout.write(lineout)


os.remove(OLD_SPEC_FILE)          
shutil.make_archive(INPUT_FILE_NAME,"zip",NUGET_TEMP_DIR)
os.rename(INPUT_FILE_NAME+".zip",INPUT_FILE_NAME)
# clean working directory
shutil.rmtree(NUGET_TEMP_DIR)