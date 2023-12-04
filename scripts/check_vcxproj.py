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


def find_missing_entries(vcxproj_path):
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

    result = {
        'ClInclude': [],
        'ClCompile': [],
    }
    for path in vcxproj_dir.iterdir():
        name, suffix = path.name.lower(), path.suffix.lower()
        if name in IGNORED_FILENAMES:
            continue
        if suffix in {".cpp"}:
            if name not in compiles:
                result['ClCompile'].append(path)
        elif suffix in {".h", ".hpp"}:
            if name not in includes:
                result['ClInclude'].append(path)

    return result


def process_file(vcxproj_path):
    result = find_missing_entries(vcxproj_path)
    ok = True
    for path in result['ClInclude']:
        print(f"{vcxproj_path}: missing ClInclude item: {path.name}", file=sys.stderr)
        ok = False
    for path in result['ClCompile']:
        print(f"{vcxproj_path}: missing ClCompile entry: {path.name}", file=sys.stderr)
        ok = False
    return ok


if __name__ == "__main__":
    arg = Path(sys.argv[1])
    if arg.is_file() and arg.suffix == '.vcxproj':
        if process_file(arg):
            sys.exit(0)
        else:
            sys.exit(1)
    elif arg.is_dir():
        exit_code = 0
        for path in arg.rglob('*.vcxproj'):
            if not process_file(path):
                exit_code = 1
        sys.exit(exit_code)
    else:
        print(f"Unsupported file: {arg}", file=sys.stderr)
        sys.exit(1)
