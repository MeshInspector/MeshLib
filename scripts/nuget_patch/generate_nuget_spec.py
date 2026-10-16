import datetime
import os
import re
import shutil
import sys
from pathlib import Path
from string import Template

WORK_DIR = Path(".")
DOTNET_DLL_DIR = WORK_DIR / sys.argv[1]
WINDOWS_X64_RUNTIME_DIR = WORK_DIR / sys.argv[2]
WINDOWS_ARM_RUNTIME_DIR = WORK_DIR / sys.argv[3]
LINUX_X64_RUNTIME_DIR = WORK_DIR / sys.argv[4]
LINUX_ARM_RUNTIME_DIR = WORK_DIR / sys.argv[5]
MACOS_X64_RUNTIME_DIR = WORK_DIR / sys.argv[6]
MACOS_ARM_RUNTIME_DIR = WORK_DIR / sys.argv[7]
VERSION = sys.argv[8][1:]

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

# For platform names, see: https://learn.microsoft.com/en-us/dotnet/core/rid-catalog
add_files( WINDOWS_X64_RUNTIME_DIR, "runtimes/win-x64/native/" )
add_files( WINDOWS_ARM_RUNTIME_DIR, "runtimes/win-arm64/native/" )
add_files( LINUX_X64_RUNTIME_DIR, "runtimes/linux-x64/native/" )
add_files( LINUX_ARM_RUNTIME_DIR, "runtimes/linux-arm64/native/" )
add_files( MACOS_X64_RUNTIME_DIR, "runtimes/osx-x64/native/" )
add_files( MACOS_ARM_RUNTIME_DIR, "runtimes/osx-arm64/native/" )

shutil.copy(WORK_DIR / "thirdparty" / "licenses" / "THIRD-PARTY-NOTICES.txt", "THIRD-PARTY-NOTICES.txt")

# the citation in the readme names the version of the package
with open(WORK_DIR / "source" / "MRDotNet2" / "readme.md", 'r', encoding='utf-8') as readme_file:
	readme = readme_file.read()
assert readme.count("{1.2.3.4}") == 1
readme = readme.replace("{1.2.3.4}", "{" + VERSION + "}")
readme = re.sub(r"(year\s*= )\{\d{4}\}", r"\g<1>{" + str(datetime.date.today().year) + "}", readme)
os.makedirs("nuget_readme", exist_ok=True)
with open(Path("nuget_readme") / "readme.md", 'w', encoding='utf-8', newline='\n') as f:
	f.write(readme)

with open(Path(__file__).parent / "template.nuspec", 'r') as template_file:
	updated_nuspec = Template(template_file.read()).substitute(
		VERSION=VERSION,
		COPYRIGHT=COPYRIGHT,
		FILES=FILES
	)
with open(SPEC_FILE, 'w') as f:
	f.write(updated_nuspec)
