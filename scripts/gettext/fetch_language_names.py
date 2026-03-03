#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pyicu",
# ]
# ///
"""
Fetch language display names from CLDR using ICU
"""
import argparse
import csv
import icu
import sys
import typing
from dataclasses import dataclass, fields


def to_sentence_case(s: str) -> str:
    """
    Capitalize the first word of the string
    """
    return s[0].upper() + s[1:]


# known languages which don't require its name to be capitalized
NO_CAPITALIZE_LANGUAGES = ['ka', 'nd', 'sn', 'tok', 'zu']


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
        if self.language not in NO_CAPITALIZE_LANGUAGES:
            self.display_name = to_sentence_case(self.display_name)
    
    def to_list(self):
        return [
            getattr(self, field.name)
            for field in fields(self)
        ]


def get_locale_infos() -> dict[str, LocaleInfo]:
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

    return locale_infos


def write_csv(out: typing.TextIO, locale_infos: dict[str, LocaleInfo]):
    csv_writer = csv.writer(out, delimiter=';', quoting=csv.QUOTE_ALL)
    csv_writer.writerows(
        [name] + info.to_list()
        for name, info in sorted(locale_infos.items())
    )


def write_cpp_map(out: typing.TextIO, locale_infos: dict[str, LocaleInfo]):
    for name, info in sorted(locale_infos.items()):
        out.write(f'{{ "{name}", "{info.display_name}" }},\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filename',
        help="output file; if output file is -, write to standard output"
    )
    parser.add_argument(
        '-f', '--format',
        choices=['csv', 'cpp-map'],
        default='csv',
        help="output format; default is csv"
    )
    args = parser.parse_args()

    if args.filename and args.filename != '-':
        open_file = lambda: open(args.filename, 'w')
    else:
        open_file = lambda: sys.stdout

    if not args.format or args.format == 'csv':
        write = write_csv
    elif args.format == 'cpp-map':
        write = write_cpp_map
    else:
        print(f"Unknown format: {args.format}")
        sys.exit(1)

    locale_infos = get_locale_infos()
    with open_file() as out:
        write(out, locale_infos)
