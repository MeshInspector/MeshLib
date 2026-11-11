#!/usr/bin/env python3
"""Check the architecture and the macOS deployment target of every packaged Mach-O binary."""

import argparse
from pathlib import Path
import re
import subprocess
import sys


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def version(value):
    parts = tuple(map(int, value.split('.')))
    return parts + (0,) * (3 - len(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('framework', type=Path)
    parser.add_argument('--arch', required=True)
    parser.add_argument(
        '--floor',
        help='highest macOS deployment target any binary may require; '
             'default: the host version',
    )
    args = parser.parse_args()
    root = args.framework.resolve(strict=True)
    host_version = output('sw_vers', '-productVersion')
    floor = args.floor or host_version
    failures = []
    count = 0
    highest = None
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open('rb') as binary:
            magic = binary.read(4)
        if magic not in (
            b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe',
            b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe',
            b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca',
            b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca',
        ):
            continue
        count += 1
        name = path.relative_to(root)
        arches = output('lipo', '-archs', str(path)).split()
        if args.arch not in arches:
            failures.append(f'{name}: missing {args.arch}; found {arches}')
            continue
        commands = output('otool', '-arch', args.arch, '-l', str(path))
        minimums = re.findall(
            r'cmd LC_VERSION_MIN_MACOSX\s+cmdsize \d+\s+version ([\d.]+)',
            commands,
        )
        minimums += re.findall(
            r'cmd LC_BUILD_VERSION\s+cmdsize \d+\s+platform (?:1|macos)\s+minos ([\d.]+)',
            commands,
        )
        if not minimums:
            failures.append(f'{name}: no macOS deployment target found')
        for minimum in minimums:
            if highest is None or version(minimum) > version(highest[0]):
                highest = (minimum, name)
            if version(minimum) > version(floor):
                failures.append(f'{name}: needs macOS {minimum}, allowed {floor} (host {host_version})')
    if not count:
        failures.append('No Mach-O binaries found')
    print(f'Checked {count} Mach-O files for {args.arch} on macOS {host_version}, floor {floor}')
    if highest:
        print(f'Highest deployment target: macOS {highest[0]} ({highest[1]})')
    for failure in failures:
        print(failure, file=sys.stderr)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
