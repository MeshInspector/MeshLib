#!/bin/bash
uv run scripts/gettext/fetch_language_names.py | awk -F ';' '{ sub("\r", "", $NF); printf("{ %s, %s },\n", $1, $NF); }' > source/MRViewer/MRLocaleNames.inl
