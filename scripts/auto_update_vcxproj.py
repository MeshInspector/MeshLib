#!/usr/bin/env python3
import codecs
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


def add_missing_entries(vcxproj_dir, item_group, tag_name, file_suffixes):
    found_files = {
        item.attrib['Include'].lower()
        for item in item_group.iterfind(f'msbuild:{tag_name}', VCXPROJ_NAMESPACES)
    }
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
                print(f"Appending {tag_name}: Include = {path}")
                item_group.append(ET.Element(
                    tag_name,
                    attrib={
                        'Include': str(path),
                    },
                ))


def fix_vcxproj(source, generated):
    """
    patch the generated XML to match the vcxproj format
    """
    # preserve BOM
    if source.startswith(codecs.BOM_UTF8):
        generated = codecs.BOM_UTF8 + generated

    # preserve trailing newline
    if source.endswith(b'\n'):
        generated += b'\n'

    # preserve newline format
    if source.find(b'\r\n') != -1:
        generated = generated.replace(b'\n', b'\r\n')

    # fix quotes
    generated = generated.replace(
        b"<?xml version='1.0' encoding='utf-8'?>",
        b'<?xml version="1.0" encoding="utf-8"?>',
        1,
    )

    # fix attribute position
    xmlns_attr = b' xmlns="http://schemas.microsoft.com/developer/msbuild/2003"'
    xmlns_attr_pos = generated.find(xmlns_attr)
    m1, m2, m3 = xmlns_attr_pos, xmlns_attr_pos + len(xmlns_attr), generated.find(b'>', xmlns_attr_pos)
    generated = generated[:m1] + generated[m2:m3] + generated[m1:m2] + generated[m3:]

    return generated


def process_file(vcxproj_path):
    with open(vcxproj_path, 'rb') as f:
        input = f.read()

    parser = ET.XMLParser(target=ET.TreeBuilder(
        insert_comments=True,
    ))
    vcxproj = ET.parse(vcxproj_path, parser)

    project = vcxproj.getroot()
    for item_group in project.iterfind('msbuild:ItemGroup', VCXPROJ_NAMESPACES):
        for tag_name, file_suffixes in KNOWN_ITEM_GROUPS.items():
            if item_group.find(f'msbuild:{tag_name}', VCXPROJ_NAMESPACES) is not None:
                add_missing_entries(vcxproj_path.parent, item_group, tag_name, file_suffixes)

    ET.register_namespace('', VCXPROJ_NAMESPACES['msbuild'])
    ET.indent(vcxproj, space="  ")
    output = ET.tostring(
        project,
        encoding='utf-8',
        xml_declaration=True,
    )
    output = fix_vcxproj(input, output)
    with open(vcxproj_path, 'wb') as f:
        f.write(output)


if __name__ == "__main__":
    arg = Path(sys.argv[1])
    if arg.is_file() and arg.suffix == '.vcxproj':
        process_file(arg)
    elif arg.is_dir():
        for path in arg.rglob('*.vcxproj'):
            process_file(path)
    else:
        print(f"Unsupported file: {arg}", file=sys.stderr)
        sys.exit(1)
