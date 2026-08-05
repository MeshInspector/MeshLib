#!/usr/bin/env python3
import argparse
import fileinput
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_gettext_command(cmd):
    if path := shutil.which(cmd):
        return path
    if gettext_root := os.getenv('GETTEXT_ROOT'):
        lookup_paths = [
            Path(gettext_root),
            Path(gettext_root) / "bin",
            ]
        lookup_path = os.pathsep.join(str(p) for p in lookup_paths)
        if path := shutil.which(cmd, path=lookup_path):
            return path
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract translatable strings from C++ sources into a .pot template.")
    parser.add_argument("pot_file", type=Path, help="output .pot file; its stem is the domain name")
    parser.add_argument("input_dir", type=Path, help="directory to scan recursively for .cpp/.h/.hpp")
    parser.add_argument("--package-name", help="value for the Project-Id-Version header field")
    args = parser.parse_args()

    pot_file = args.pot_file
    input_dir = args.input_dir

    xgettext = find_gettext_command('xgettext')
    if not xgettext:
        print(
            "Cannot find xgettext. Set GETTEXT_ROOT environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    domain_name = pot_file.stem

    input_files = list(map(str, itertools.chain(
        input_dir.rglob("*.cpp"),
        input_dir.rglob("*.h"),
        input_dir.rglob("*.hpp"),
    )))
    input_files.sort()

    with open(Path(__file__).parent / "xgettext_options.txt") as options_file:
        xgettext_options = [
            line.strip()
            for line in options_file
        ]
    if args.package_name:
        xgettext_options.append(f"--package-name={args.package_name}")
    # use temporary file to bypass Windows' command line length limitations
    with tempfile.NamedTemporaryFile('w', delete_on_close=False) as input_list:
        input_list.writelines(line + "\n" for line in input_files)
        input_list.close()

        print(f"Extracting strings ...")

        subprocess.run([
            xgettext,
            *xgettext_options,
            f"--default-domain={domain_name}",
            f"--output={pot_file}",
            f"--files-from={input_list.name}",
        ], check=True)

    # force set UTF-8 charset
    with fileinput.FileInput(pot_file, inplace=True) as f:
        for line in f:
            line = line.replace("Content-Type: text/plain; charset=CHARSET", "Content-Type: text/plain; charset=UTF-8")
            print(line, end='')

    # TODO: parse .items.json files
