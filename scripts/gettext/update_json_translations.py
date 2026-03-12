#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


POT_HEADER = '''#, fuzzy
msgid ""
msgstr ""
"Language: \\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"

'''


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
        print("Usage: update_json_translations.py POT_FILE INPUT_JSON")
        sys.exit(0)

    _, pot_file, input_json = sys.argv
    pot_file = Path(pot_file)
    input_json = Path(input_json)

    msgmerge = find_gettext_command('msgmerge')
    if not msgmerge:
        print(
            "Cannot find msgmerge. Set GETTEXT_ROOT environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = pot_file.parent
    domain_name = pot_file.stem

    records = []
    with open(input_json, 'r') as f:
        doc = json.load(f)
        for item in doc['Items']:
            # TODO: support line numbers
            if 'Name' not in item:
                continue
            if 'Caption' in item:
                records.append(item['Caption'])
            else:
                records.append(item['Name'])
            if 'Tooltip' in item:
                records.append(item['Tooltip'])

    with open(pot_file, 'w') as f:
        f.write(POT_HEADER)
        for rec in records:
            # TODO: support line numbers
            #f.write(f"#: {input_json}:{rec.lineno}\n")
            f.write(f'msgid "{rec}"\n')
            f.write('msgstr ""\n')
            f.write('\n')

    for po_file in output_dir.glob(f"*/{domain_name}.po"):
        locale_name = po_file.parent.name
        print(f"Updating {locale_name} locale ...")

        subprocess.run([
            msgmerge,
            "--update",
            po_file,
            pot_file,
        ], check=True)
