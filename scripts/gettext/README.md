This directory contains helper scripts for the localization support.

- update_translations.py extracts translatable strings from the source files to .pot file and updates the existing .po files (add/remove records).
- compile_translations.py converts the existing .po files to .mo files that can be distributed and used by apps.
- fetch_language_names.py generates a list of names for known locales, including a language name, a country name, and a script name, if available.
