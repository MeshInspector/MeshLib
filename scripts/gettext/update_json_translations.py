#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


POT_HEADER = '''#, fuzzy
msgid ""
msgstr ""
"Language: \\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"

'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract translatable strings from ribbon .items.json / .ui.json into a .pot template.")
    parser.add_argument("pot_file", type=Path, help="output .pot file; its stem is the domain name")
    parser.add_argument("items_json", type=Path, help=".items.json holding item captions and tooltips")
    parser.add_argument("ui_json", type=Path, nargs='?', help=".ui.json holding tab names; defaults to the sibling of ITEMS_JSON")
    parser.add_argument("--package-name", help="value for the Project-Id-Version header field")
    args = parser.parse_args()

    pot_file = args.pot_file
    input_json = args.items_json
    # Auto-detect paired .ui.json if not provided explicitly
    if args.ui_json is not None:
        ui_json = args.ui_json
    else:
        ui_json_path = input_json.with_suffix('').with_suffix('.ui.json')
        ui_json = ui_json_path if ui_json_path.exists() else None

    domain_name = pot_file.stem

    # Contextless records from .items.json (captions, tooltips)
    records = []
    def add_record(s):
        if s and s not in records:
            records.append(s)
    with open(input_json, 'r') as f:
        doc = json.load(f)
        for item in doc['Items']:
            # TODO: support line numbers
            if 'Name' not in item:
                continue
            if 'Caption' in item:
                add_record(item['Caption'])
            else:
                add_record(item['Name'])
            if 'Tooltip' in item:
                add_record(item['Tooltip'])

    # Tab name records from .ui.json (with "Tab name" context)
    tab_name_records = []
    if ui_json is not None:
        with open(ui_json, 'r') as f:
            ui_doc = json.load(f)
            tab_name_records = [name for tab in ui_doc.get('Tabs', []) if (name := tab.get('Name'))]

    if args.package_name:
        POT_HEADER = POT_HEADER[:-1] + f'"Project-Id-Version: {args.package_name}\\n"\n' + POT_HEADER[-1:]

    with open(pot_file, 'w') as f:
        f.write(POT_HEADER)
        for rec in records:
            rec = rec.replace('\n', "\\n")
            # TODO: support line numbers
            #f.write(f"#: {input_json}:{rec.lineno}\n")
            f.write(f'msgid "{rec}"\n')
            f.write('msgstr ""\n')
            f.write('\n')
        for name in tab_name_records:
            f.write('msgctxt "Tab name"\n')
            f.write(f'msgid "{name}"\n')
            f.write('msgstr ""\n')
            f.write('\n')

