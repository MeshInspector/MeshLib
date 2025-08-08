#!/usr/bin/env python3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

VCXPROJ_NAMESPACES = {
    'msbuild': "http://schemas.microsoft.com/developer/msbuild/2003",
}

KNOWN_ITEM_GROUPS = {
    'ClCompile': {".c", ".cpp"},
    'ClInclude': {".h", ".hpp", ".cuh"},
    'CudaCompile': {".cu"},
}

IGNORED_FILENAMES = {
    # CMake-specific files
    "config.h",
    "config_cmake.h",
    # macOS-specific files
    "mrfiledialogcocoa.h",
    "mrtouchpadcocoahandler.h",
}


def find_missing_entries(vcxproj_path):
    vcxproj_dir = vcxproj_path.parent

    vcxproj = ET.parse(vcxproj_path)
    project = vcxproj.getroot() 

    result = {}
 
    for tag_name, file_suffixes in KNOWN_ITEM_GROUPS.items():
        found_files = {
            item.attrib['Include'].lower()
            for item in project.iterfind(
                f'msbuild:ItemGroup/msbuild:{tag_name}',
                VCXPROJ_NAMESPACES,
            )
        }
        results = []
        for root, dirs, files in vcxproj_dir.walk():
            dir = root.relative_to(vcxproj_dir)
            for filename in files:
                path = PureWindowsPath(dir / filename)
                name, suffix = path.name.lower(), path.suffix.lower()
                if name in IGNORED_FILENAMES:
                    continue
                if suffix not in file_suffixes:
                    continue
                if str(path).lower() not in found_files:
                    results.append(path)
        result[tag_name] = results

    return result


def process_file(vcxproj_path):
    result = find_missing_entries(vcxproj_path)
    ok = True
    for group_name, group in result.items():
        for path in group:
            print(f"{vcxproj_path}: missing {group_name} item: {path}", file=sys.stderr)
            ok = False
    return ok


if __name__ == "__main__":
    exit_code = 0
    queue=[]
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            queue += [*path.rglob('*.vcxproj')]
        elif path.is_file() and path.suffix == '.vcxproj':
            queue += [path]
        else:
            print(f"Unsupported file: {arg}", file=sys.stderr)
            exit_code = 1
    for path in queue:
        if not process_file(path):
            exit_code = 1
    sys.exit(exit_code)
