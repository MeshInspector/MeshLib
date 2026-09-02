"""Print each candidate's prologue from a dumpbin /disasm listing and flag the
defect: `bl __chkstk` reached before the link register is saved.

dumpbin prints the fp/lr aliases, not x29/x30, so match both spellings.

Usage: python arm64_prologue_report.py repro.asm
"""

import re
import sys

LR_SAVE = re.compile(r"\b(str|stp)\b.*\b(x30|lr)\b")
LR_LOAD = re.compile(r"\b(ldr|ldp)\b.*\b(x30|lr)\b")
CHKSTK = re.compile(r"chkstk|alloca_probe")


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


def analyse(body):
    frame = ""
    chkstk_at = lr_save_at = None
    for i, line in enumerate(body):
        low = line.lower()
        m = re.search(r"sub\s+sp,sp,#(0x[0-9a-f]+)", low)
        if m and not frame:
            frame = m.group(1)
        if chkstk_at is None and CHKSTK.search(low):
            chkstk_at = i
        if lr_save_at is None and LR_SAVE.search(low):
            lr_save_at = i
    if chkstk_at is None:
        return "no-probe", frame, chkstk_at, lr_save_at
    if lr_save_at is None:
        return "BAD (lr never saved)", frame, chkstk_at, lr_save_at
    if chkstk_at < lr_save_at:
        return "BAD (chkstk destroys lr first)", frame, chkstk_at, lr_save_at
    return "ok", frame, chkstk_at, lr_save_at


def main():
    lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
    for name, body in functions(lines):
        if not re.match(r"^v\d+$", name):
            continue
        state, frame, chkstk_at, lr_save_at = analyse(body)
        print(
            "  %-4s %-32s frame=%-6s chkstk@%s lr-save@%s"
            % (name, state, frame or "-", chkstk_at, lr_save_at)
        )
        if state.startswith("BAD"):
            for i, line in enumerate(body):
                text = line.strip()
                keep = i < 6 or CHKSTK.search(text.lower()) or LR_SAVE.search(
                    text.lower()
                ) or LR_LOAD.search(text.lower()) or "ret" in text.lower()
                if text and keep:
                    print("         " + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
