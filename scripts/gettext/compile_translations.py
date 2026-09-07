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
    if len(sys.argv) != 3:
        print("Usage: compile_translations.py INPUT_DIR OUTPUT_DIR")
        sys.exit(0)

    _, input_dir, output_dir = sys.argv
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    msgfmt = find_gettext_command('msgfmt')
    if not msgfmt:
        print(
            "Cannot find msgfmt. Set GETTEXT_ROOT environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)
    
    for po_file in input_dir.glob("*/*.po"):
        locale_name = po_file.parent.name
        domain_name = po_file.stem
        print(f"Generating {locale_name} locale for {domain_name} ...")

        mo_output_dir = output_dir / locale_name / "LC_MESSAGES"
        if not mo_output_dir.exists():
            mo_output_dir.mkdir(parents=True)
        
        mo_output_file = mo_output_dir / f"{domain_name}.mo"
        subprocess.run([
            msgfmt,
            po_file,
            f"--output-file={mo_output_file}",
            "--check",
        ], check=True)
