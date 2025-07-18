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
    "mrfiledialogcocoa.h",
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
    cuda_compiles = {
        item.attrib['Include'].lower()
        for item in project.iterfind(
            'msbuild:ItemGroup/msbuild:CudaCompile',
            VCXPROJ_NAMESPACES,
        )
    }

    result = {
        'ClInclude': [],
        'ClCompile': [],
        'CudaCompile': [],
    }
    for path in vcxproj_dir.iterdir():
        name, suffix = path.name.lower(), path.suffix.lower()
        if name in IGNORED_FILENAMES:
            continue
        if suffix in {".cpp"}:
            if name not in compiles:
                result['ClCompile'].append(path)
        elif suffix in {".h", ".hpp", ".cuh"}:
            if name not in includes:
                result['ClInclude'].append(path)
        elif suffix in {".cu"}:
            if name not in cuda_compiles:
                result['CudaCompile'].append(path)

    return result


def process_file(vcxproj_path):
    result = find_missing_entries(vcxproj_path)
    ok = True
    for group_name, group in result.items():
        for path in group:
            print(f"{vcxproj_path}: missing {group_name} item: {path.name}", file=sys.stderr)
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
