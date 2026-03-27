#!/usr/bin/env python3
import json
import sys
from pathlib import Path


POT_HEADER = '''#, fuzzy
msgid ""
msgstr ""
"Language: \\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"

'''


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: update_json_translations.py POT_FILE ITEMS_JSON [UI_JSON]")
        sys.exit(0)

    pot_file = Path(sys.argv[1])
    input_json = Path(sys.argv[2])
    # Auto-detect paired .ui.json if not provided explicitly
    if len(sys.argv) == 4:
        ui_json = Path(sys.argv[3])
    else:
        ui_json_path = input_json.with_suffix('').with_suffix('.ui.json')
        ui_json = ui_json_path if ui_json_path.exists() else None

    domain_name = pot_file.stem

    # Contextless records from .items.json (captions, tooltips)
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

    # Tab name records from .ui.json (with "Tab name" context)
    tab_name_records = []
    if ui_json is not None:
        with open(ui_json, 'r') as f:
            ui_doc = json.load(f)
            tab_name_records = [name for tab in ui_doc.get('Tabs', []) if (name := tab.get('Name'))]

    with open(pot_file, 'w') as f:
        f.write(POT_HEADER)
        for rec in records:
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

