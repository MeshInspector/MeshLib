import os
import shutil
import subprocess
import sys
from pathlib import Path


def try_joinpath(path: str | None, *segments: list[str]):
    return Path(path).joinpath(*segments) if path else None


def find_msgfmt():
    msgfmt_exe = os.getenv('MSGFMT_EXECUTABLE')
    gettext_root = os.getenv('GETTEXT_ROOT')
    candidates = [
        shutil.which("msgfmt.exe"),
        msgfmt_exe,
        try_joinpath(gettext_root, "msgfmt.exe"),
        try_joinpath(gettext_root, "bin", "msgfmt.exe"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    else:
        return None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: win_update_translations.py INPUT_DIR OUTPUT_DIR")
        sys.exit(0)

    _, input_dir, output_dir = sys.argv
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    msgfmt_exe = find_msgfmt()
    if not msgfmt_exe:
        print(
            "Cannot find msgfmt executable. Set MSGFMT_EXECUTABLE or GETTEXT_ROOT environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)
    
    for po_file in input_dir.glob("*/*.po"):
        locale_name = po_file.parent.name
        domain_name = po_file.stem

        mo_output_dir = output_dir / locale_name / "LC_MESSAGES"
        if not mo_output_dir.exists():
            mo_output_dir.mkdir(parents=True)
        
        mo_output_file = mo_output_dir / f"{domain_name}.po"
        subprocess.run([
            msgfmt_exe,
            po_file,
            f"--output-file={mo_output_file}",
            "--check",
        ], check=True)
