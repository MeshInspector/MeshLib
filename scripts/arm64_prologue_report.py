"""Print each function's prologue from a dumpbin /disasm listing and flag the
defect: `bl __chkstk` reached before x30 is stored.

Usage: python arm64_prologue_report.py repro.asm
"""

import re
import sys


def functions(lines):
    name, body = None, []
    for line in lines:
        m = re.match(r"^([A-Za-z_$?][\w$?@]*):$", line.strip())
        if m:
            if name:
                yield name, body
            name, body = m.group(1), []
        elif name is not None:
            body.append(line.rstrip())
    if name:
        yield name, body


def verdict(body):
    """Walk the prologue; report which of __chkstk / x30-store comes first."""
    for line in body[:40]:
        low = line.lower()
        if "chkstk" in low or "alloca_probe" in low:
            return "BAD  - bl __chkstk before x30 is saved"
        if re.search(r"\b(str|stp)\b.*\bx30\b", low):
            return "ok   - x30 saved first"
    return "n/a  - neither in the first 40 instructions"


def main():
    lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
    for name, body in functions(lines):
        if not re.match(r"^v\d+$", name):
            continue
        print("  %-4s %s" % (name, verdict(body)))
        for line in body[:8]:
            text = line.strip()
            if text:
                print("         " + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
