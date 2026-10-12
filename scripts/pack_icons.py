#!/usr/bin/env python3
"""Pack the icon folders of a resource directory into <resource_dir>/icons.zip and delete them.

MR::RibbonIcons reads that one file at startup instead of a thousand small ones.
Entries are named like "icons/X1/Tool.png", relative to the resource directory.

usage: pack_icons.py <resource_dir>
"""
import pathlib
import shutil
import sys
import zipfile

ICON_FOLDERS = ['icons', 'object_icons', 'independent_icons', 'logos']


def main():
    resource_dir = pathlib.Path(sys.argv[1])
    folders = [resource_dir / name for name in ICON_FOLDERS if (resource_dir / name).is_dir()]
    if not folders:
        sys.exit(f'no icon folders found in {resource_dir}')

    zip_path = resource_dir / 'icons.zip'
    count = 0
    # PNG is already compressed: storing the entries makes both packing and loading cheaper
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
        for folder in folders:
            for file in sorted(folder.rglob('*')):
                if file.is_file() and file.suffix.lower() == '.png':
                    z.write(file, file.relative_to(resource_dir).as_posix())
                    count += 1
    for folder in folders:
        shutil.rmtree(folder)
    print(f'{zip_path}: {count} icons from {", ".join(f.name for f in folders)}')


if __name__ == '__main__':
    main()
