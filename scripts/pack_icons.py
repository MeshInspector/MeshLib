#!/usr/bin/env python3
"""Pack the icon folders of resource directories into one zip, which MR::RibbonIcons reads on start
instead of a thousand small files.

Entries are named like "icons/X1/Tool.png", relative to the resource directory the icon comes from.

usage: pack_icons.py <output zip> <resource_dir>...
"""
import pathlib
import sys
import zipfile

ICON_FOLDERS = ['icons', 'object_icons', 'independent_icons', 'logos']


def main():
    zip_path = pathlib.Path(sys.argv[1])
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # PNG is already compressed: storing the entries makes both packing and loading cheaper
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
        for resource_dir in map(pathlib.Path, sys.argv[2:]):
            for folder in ICON_FOLDERS:
                for file in sorted((resource_dir / folder).rglob('*')):
                    if file.is_file() and file.suffix.lower() == '.png':
                        z.write(file, file.relative_to(resource_dir).as_posix())
                        count += 1
    print(f'{zip_path}: {count} icons')


if __name__ == '__main__':
    main()
