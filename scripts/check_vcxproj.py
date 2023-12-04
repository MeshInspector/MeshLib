#!/usr/bin/env python3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

VCXPROJ_NAMESPACES = {
    'msbuild': "http://schemas.microsoft.com/developer/msbuild/2003",
}

IGNORED_FILENAMES = {
    # CMake-specific files
    "config.h",
    "config_cmake.h",
    # macOS-specific files
    "mrtouchpadcocoahandler.h",
}

if __name__ == "__main__":
    vcxproj_path = Path(sys.argv[1])
    vcxproj_dir = vcxproj_path.parent

    vcxproj = ET.parse(vcxproj_path)
    project = vcxproj.getroot()

    includes = {
        item.attrib['Include'].lower()
        for item in project.iterfind(
            'msbuild:ItemGroup/msbuild:ClInclude',
            VCXPROJ_NAMESPACES,
        )
    }
    compiles = {
        item.attrib['Include'].lower()
        for item in project.iterfind(
            'msbuild:ItemGroup/msbuild:ClCompile',
            VCXPROJ_NAMESPACES,
        )
    }

    exit_code = 0
    for path in vcxproj_dir.iterdir():
        name, suffix = path.name.lower(), path.suffix.lower()
        if name in IGNORED_FILENAMES:
            continue
        if suffix in {".cpp"}:
            if name not in compiles:
                print(f"{vcxproj_path}: missing ClCompile item: {path.name}", file=sys.stderr)
                exit_code = 1
        elif suffix in {".h", ".hpp"}:
            if name not in includes:
                print(f"{vcxproj_path}: missing ClInclude item: {path.name}", file=sys.stderr)
                exit_code = 1

    sys.exit(exit_code)
