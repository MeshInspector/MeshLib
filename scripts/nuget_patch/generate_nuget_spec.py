import os
import shutil
import sys
from pathlib import Path

WORK_DIR = Path(".")
DOTNET_DLL_DIR = WORK_DIR / sys.argv[1]
WINDOWS_RUNTIME_DIR = WORK_DIR / sys.argv[2]
LINUX_RUNTIME_DIR = WORK_DIR / sys.argv[3]
VERSION = sys.argv[4][1:]

SPEC_FILE = WORK_DIR / "Package.nuspec"
LICENSE_FILE = WORK_DIR / "LICENSE"
shutil.copy(LICENSE_FILE, str(LICENSE_FILE + ".txt") )
LICENSE_FILE = Path( str(LICENSE_FILE + ".txt") )

COPYRIGHT_FILE = WORK_DIR / "scripts" / "copyright_header.txt"
COPYRIGHT = open(COPYRIGHT_FILE,'r').read()[3:]

f = open(SPEC_FILE, 'w')
f.write('<?xml version="1.0" encoding="utf-8"?>\n')
f.write('<package>\n')
f.write('\t<metadata>\n')
f.write('\t\t<id>MeshLib</id>\n')
f.write('\t\t<version>' + VERSION + '</version>\n')
f.write('\t\t<authors>AMV Consulting</authors>\n')
f.write('\t\t<owners>AMV Consulting</owners>\n')
f.write('\t\t<projectUrl>https://meshlib.io</projectUrl>\n')
f.write('\t\t<description>Mesh processing library</description>\n')
f.write('\t\t<releaseNotes>https://github.com/MeshInspector/MeshLib/releases</releaseNotes>\n')
f.write('\t\t<copyright>' + COPYRIGHT + '</copyright>\n')
f.write('\t\t<dependencies>\n')
f.write('\t\t\t<group targetFramework="netstandard2.0"/>\n')
f.write('\t\t</dependencies>\n')
f.write('\t\t<icon>images/MeshInspector_icon.png</icon>\n')
f.write('\t\t<license type="file">' + str(LICENSE_FILE.name) + '</license>\n')
f.write('\t\t<readme>docs/readme_dotnet.md</readme>\n')
f.write('\t</metadata>\n')

f.write('\t<files>\n')
f.write('\t\t<file src="./macos/MeshInspector_icon.png" target="images/"></file>\n')
f.write('\t\t<file src="' + str(LICENSE_FILE) +'" target=""></file>\n')
f.write('\t\t<file src="./readme_dotnet.md" target="docs/"></file>\n')

def add_files(folder : Path, target):
	for address, dirs, files in folder:
		for file in files:
			fname = Path(file)
			if not fname.is_file():
				continue
			f.write( '\t\t<file src="' + str(folder / file) +  '" target="' + target + '""></file>\n' )

add_files( DOTNET_DLL_DIR, "lib/netstandard2.0/" )
add_files( WINDOWS_RUNTIME_DIR, "runtimes/win-x64/native/" )
add_files( LINUX_RUNTIME_DIR, "runtimes/linux-x64/native/" )

f.write('\t</files>\n')
f.write('</package>\n')
f.close()
