#!/usr/bin/env python3
"""Bundle Homebrew dylib dependencies into a MeshLib.framework version dir.

Walks all Mach-O files under <framework_version_dir>/bin and /lib, copies any
dependency that resolves under a Homebrew prefix into <framework>/lib, and
rewrites the LC_LOAD_DYLIB / LC_ID_DYLIB entries to @rpath/<basename>. An
LC_RPATH is added so the bundled libs are found relative to the binary:
  - executables in bin/ get @executable_path/../lib
  - dylibs in lib/ (incl. just-bundled ones) get @loader_path/.

System libraries (/usr/lib, /System) and libpython* are intentionally left as
external references.

This makes the produced .pkg robust against Homebrew bottle SONAME drift
(e.g. jsoncpp dropping libjsoncpp.27.dylib alias) without forcing version pins
on every dep in requirements/macos.txt.
"""
from __future__ import annotations

import argparse
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

HOMEBREW_PREFIXES = ("/usr/local/", "/opt/homebrew/")
SYSTEM_PREFIXES = ("/usr/lib/", "/System/")
RELATIVE_PREFIXES = ("@rpath/", "@loader_path/", "@executable_path/")
# Leave these to the host system / Homebrew
SKIP_BASENAME_RE = re.compile(r"^(libpython|Python$)")

MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",  # MH_MAGIC_64 / _CIGAM_64
    b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xce",  # MH_MAGIC / _CIGAM
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",  # FAT_MAGIC / _CIGAM
    b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",  # FAT_MAGIC_64 / _CIGAM_64
}


def log(msg: str) -> None:
    print(f"[bundle-dylibs] {msg}", file=sys.stderr)


def is_macho(p: Path) -> bool:
    if p.is_symlink() or not p.is_file():
        return False
    try:
        with p.open("rb") as f:
            return f.read(4) in MACHO_MAGICS
    except OSError:
        return False


def collect_machos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if is_macho(p))


def otool_L(p: Path) -> list[str]:
    out = subprocess.check_output(["otool", "-L", str(p)], text=True)
    deps: list[str] = []
    for line in out.splitlines()[1:]:
        m = re.match(r"\s+(\S+) \(", line)
        if m:
            deps.append(m.group(1))
    return deps


def should_bundle(load_path: str) -> bool:
    if load_path.startswith(RELATIVE_PREFIXES):
        return False
    if load_path.startswith(SYSTEM_PREFIXES):
        return False
    if not load_path.startswith(HOMEBREW_PREFIXES):
        return False
    if SKIP_BASENAME_RE.match(Path(load_path).name):
        return False
    return True


def make_writable(p: Path) -> None:
    p.chmod(p.stat().st_mode | stat.S_IWUSR)


def install_name_tool(*args: str) -> None:
    subprocess.check_call(["install_name_tool", *args])


def codesign_adhoc(p: Path) -> None:
    subprocess.check_call([
        "codesign", "--force", "--sign", "-",
        "--preserve-metadata=entitlements,requirements,flags,runtime",
        str(p),
    ])


def has_rpath(p: Path, rpath: str) -> bool:
    out = subprocess.check_output(["otool", "-l", str(p)], text=True)
    # Lines look like:    path @executable_path/../lib (offset 12)
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("path ") and s[5:].startswith(rpath + " "):
            return True
    return False


def bundle(framework_dir: Path) -> None:
    bin_dir = framework_dir / "bin"
    lib_dir = framework_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)

    seeds = collect_machos(bin_dir) + collect_machos(lib_dir)
    log(f"seed mach-o files: {len(seeds)}")

    # BFS: copy every Homebrew dep transitively into lib_dir
    queue: list[Path] = list(seeds)
    bundled: dict[str, Path] = {}
    visited: set[Path] = set()
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        for dep in otool_L(cur):
            if not should_bundle(dep):
                continue
            name = Path(dep).name
            if name in bundled:
                continue
            src = Path(dep).resolve()
            if not src.exists():
                # The very problem we're guarding against: link-time path no
                # longer exists in the current bottle. Try the basename via
                # /usr/local/lib or /opt/homebrew/lib as a last resort.
                fallback = None
                for pref in HOMEBREW_PREFIXES:
                    cand = Path(pref) / "lib" / name
                    if cand.exists():
                        fallback = cand.resolve()
                        break
                if fallback is None:
                    log(f"WARN: cannot resolve {dep}; skipping")
                    continue
                src = fallback
            dst = lib_dir / name
            log(f"copy {src} -> {dst}")
            shutil.copy2(src, dst, follow_symlinks=True)
            make_writable(dst)
            bundled[name] = dst
            queue.append(dst)

    # Rewrite ids, load commands, and add rpaths on every Mach-O we ship
    all_files = collect_machos(bin_dir) + collect_machos(lib_dir)
    for p in all_files:
        make_writable(p)
        is_dylib = p.suffix == ".dylib" or p.is_relative_to(lib_dir)
        if is_dylib:
            install_name_tool("-id", f"@rpath/{p.name}", str(p))

        for dep in otool_L(p):
            if not should_bundle(dep):
                continue
            install_name_tool(
                "-change", dep, f"@rpath/{Path(dep).name}", str(p),
            )

        rpath = "@executable_path/../lib" if p.is_relative_to(bin_dir) else "@loader_path/."
        if not has_rpath(p, rpath):
            install_name_tool("-add_rpath", rpath, str(p))

        codesign_adhoc(p)

    log(f"bundled {len(bundled)} dylibs into {lib_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "framework_version_dir",
        type=Path,
        help="Path to MeshLib.framework/Versions/<X.Y.Z.W>",
    )
    args = ap.parse_args()
    bundle(args.framework_version_dir.resolve())


if __name__ == "__main__":
    main()
