import os
import shutil
import sys
from pathlib import Path
from string import Template

WORK_DIR = Path(".")
DOTNET_DLL_DIR = WORK_DIR / sys.argv[1]
WINDOWS_RUNTIME_DIR = WORK_DIR / sys.argv[2]
LINUX_RUNTIME_DIR = WORK_DIR / sys.argv[3]
VERSION = sys.argv[4][1:]

SPEC_FILE = WORK_DIR / "Package.nuspec"
LICENSE_FILE = WORK_DIR / "LICENSE"
shutil.copy(LICENSE_FILE, "LICENSE.txt" )

COPYRIGHT_FILE = WORK_DIR / "scripts" / "copyright_header.txt"
with open(COPYRIGHT_FILE,'r') as copyright_file:
	COPYRIGHT = copyright_file.read()[3:]

def add_files( folder : Path, target ):
	global FILES
	for address, dirs, files in os.walk(folder):
		for file in files:
			fname = Path(folder/file)
			if not fname.is_file():
				continue
			FILES += f'\t\t<file src="{str(fname)}" target="{target}"></file>\n'

FILES = ""
add_files( DOTNET_DLL_DIR, "lib/netstandard2.0/" )
add_files( WINDOWS_RUNTIME_DIR, "runtimes/win-x64/native/" )
add_files( LINUX_RUNTIME_DIR, "runtimes/linux-x64/native/" )

with open(Path(__file__).parent / "template.nuspec", 'r') as template_file:
	updated_nuspec = Template(template_file.read()).substitute(
		VERSION=VERSION,
		COPYRIGHT=COPYRIGHT,
		FILES=FILES
	)
with open(SPEC_FILE, 'w') as f:
	f.write(updated_nuspec)
