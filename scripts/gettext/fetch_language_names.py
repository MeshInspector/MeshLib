#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pyicu",
# ]
# ///
"""
Fetch language display names from CLDR using ICU
"""
import csv
import icu
import sys
from dataclasses import dataclass, fields


@dataclass
class LocaleInfo:
    language: str = ""
    script: str = ""
    territory: str = ""
    variant: str = ""
    display_name: str = ""

    def __init__(self, loc: icu.Locale):
        self.language = loc.getLanguage()
        self.script = loc.getScript()
        self.territory = loc.getCountry()
        self.variant = loc.getVariant()
        self.display_name = loc.getDisplayName(loc)
    
    def to_list(self):
        return [
            getattr(self, field.name)
            for field in fields(self)
        ]


if __name__ == "__main__":
    available_locales = icu.Locale.getAvailableLocales()
    locale_infos = {
        name: LocaleInfo(loc)
        for name, loc in available_locales.items()
    }
    # find script-neutral territory locales
    lang_territories = set(
        f"{info.language}_{info.territory}"
        for info in locale_infos.values()
        if info.territory
    )
    unavailable_lang_territories = lang_territories - set(available_locales.keys())
    locale_infos.update({
        name: LocaleInfo(icu.Locale(name))
        for name in unavailable_lang_territories
    })
    # output to CSV
    csv_writer = csv.writer(sys.stdout, delimiter=';', quoting=csv.QUOTE_ALL)
    csv_writer.writerows(
        [name] + info.to_list()
        for name, info in sorted(locale_infos.items())
    )
