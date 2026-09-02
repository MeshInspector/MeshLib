"""Delta-debug a C file down to the smallest form that still makes MSVC ARM64
emit `bl __chkstk` ahead of the link-register spill at /O2 /Gs0.

llvm-reduce operates on LLVM IR for a clang pipeline, so it cannot drive MSVC's
code generator; C-Reduce and cvise are not available on Windows ARM64. This is
the same idea with `cl` + `dumpbin` as the interestingness test: ddmin over
lines, then a numeric-shrink pass, then ddmin again.

Usage: python arm64_reduce.py input.c [max_tests]
"""

import os
import re
import subprocess
import sys

CHKSTK = re.compile(r"chkstk|alloca_probe")
LR_SAVE = re.compile(r"\b(str|stp)\b.*\b(x30|lr)\b")
FUNC = re.compile(r"^([A-Za-z_$?][\w$?@]*):$")

tests = 0
max_tests = 800


def interesting(text):
    """True when the file compiles and some function spills lr after a chkstk call."""
    global tests
    tests += 1
    with open("cand.c", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    for stale in ("cand.obj", "cand.asm"):
        if os.path.exists(stale):
            os.remove(stale)
    r = subprocess.run(
        ["cl", "/nologo", "/c", "/O2", "/Gs0", "/Fo:cand.obj", "cand.c"],
        capture_output=True,
    )
    if r.returncode != 0 or not os.path.exists("cand.obj"):
        return False
    r = subprocess.run(
        ["dumpbin", "/nologo", "/disasm:nobytes", "cand.obj"], capture_output=True
    )
    if r.returncode != 0:
        return False
    name, chk, lrs = None, None, None
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        m = FUNC.match(line.strip())
        if m:
            if name and chk is not None and (lrs is None or chk < lrs):
                return True
            name, chk, lrs = m.group(1), None, None
            idx = 0
            continue
        if name is None:
            continue
        low = line.lower()
        if CHKSTK.search(low) and chk is None:
            chk = idx
        if LR_SAVE.search(low) and lrs is None:
            lrs = idx
        idx += 1
    return bool(name and chk is not None and (lrs is None or chk < lrs))


def ddmin(lines):
    """Classic ddmin over lines: drop chunks, keep whatever stays interesting."""
    n = max(len(lines) // 2, 1)
    while n >= 1:
        i = 0
        while i < len(lines):
            trial = lines[:i] + lines[i + n :]
            if tests < max_tests and trial and interesting("".join(trial)):
                lines = trial
                print("    -%d lines -> %d left" % (n, len(lines)), flush=True)
            else:
                i += n
        if n == 1:
            break
        n = max(n // 2, 1)
    return lines


def shrink_numbers(text):
    """Try smaller values for every literal; frame layout may not need the big ones."""
    for lit in sorted(set(re.findall(r"0x[0-9a-fA-F]+|\b\d{2,}\b", text)), key=len, reverse=True):
        for repl in ("0x8", "2", "1"):
            if lit == repl or tests >= max_tests:
                continue
            trial = text.replace(lit, repl)
            if trial != text and interesting(trial):
                print("    %s -> %s" % (lit, repl), flush=True)
                text = trial
                break
    return text


def main():
    src = open(sys.argv[1], encoding="utf-8").read()
    global max_tests
    if len(sys.argv) > 2:
        max_tests = int(sys.argv[2])

    if not interesting(src):
        print("input is NOT interesting - nothing to reduce", flush=True)
        return 1
    print("input reproduces; reducing (budget %d compiles)" % max_tests, flush=True)

    lines = ddmin(src.splitlines(keepends=True))
    text = shrink_numbers("".join(lines))
    text = "".join(ddmin(text.splitlines(keepends=True)))

    print("\n==== compiles used: %d" % tests, flush=True)
    print("==== reduced source ====", flush=True)
    print(text, flush=True)
    open("reduced.c", "w", encoding="utf-8", newline="\n").write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
