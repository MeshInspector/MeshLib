#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
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
    if len(sys.argv) < 3:
        print("Usage: merge_po_from_pot.py PO_DIR POT_FILE [POT_FILE ...]")
        sys.exit(0)

    _, po_dir, *pot_files = sys.argv
    po_dir = Path(po_dir)

    msgmerge = find_gettext_command('msgmerge')
    if not msgmerge:
        print(
            "Cannot find msgmerge. Set GETTEXT_ROOT environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    for pot_file in pot_files:
        pot_file = Path(pot_file)
        domain_name = pot_file.stem

        for po_file in po_dir.glob(f"*/{domain_name}.po"):
            locale_name = po_file.parent.name
            print(f"Updating {locale_name} locale for {domain_name} ...")

            subprocess.run([
                msgmerge,
                "--update",
                po_file,
                pot_file,
            ], check=True)
