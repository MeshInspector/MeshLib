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

Why not dylibbundler / CMake BundleUtilities:
  - The dst basename must match the *referrer's* load-command name, not the
    realpath after symlink resolution. e.g. binaries reference
    @rpath/libglfw.3.dylib while the Cellar's real file is libglfw.3.4.dylib
    -- dylibbundler stores the realpath's basename, breaking dyld lookup.
    Same shape for fmt / tbb / spdlog / openvdb / opencascade libTK*.
  - The active Homebrew prefix is detected at runtime via `brew --prefix`
    (the arm64 self-hosted runner installs Homebrew at /Users/runner/.homebrew,
    not /usr/local or /opt/homebrew); off-the-shelf tools hardcode the two
    standard prefixes and silently bundle nothing on that runner.
  - Intra-bottle @rpath siblings (e.g. gdcm's libgdcmMEXD ->
    @rpath/libsocketxx.1.2.dylib at @loader_path/.) are resolved against the
    *source* directory of the lib that just got bundled, so we don't depend on
    Homebrew's rpath embedding to keep working.
  - No new build-time tool to install on every macOS runner; otool /
    install_name_tool / codesign ship with Xcode CLT, which is already
    required for the existing build.
"""
from __future__ import annotations

import argparse
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

def _detect_homebrew_prefixes() -> tuple[str, ...]:
    """Standard locations + whatever `brew --prefix` reports.

    Self-hosted runners may install Homebrew under a custom prefix
    (e.g. /Users/runner/.homebrew on the arm64 build runner). Hardcoding
    only /usr/local and /opt/homebrew silently disables bundling there.
    """
    defaults = ("/usr/local/", "/opt/homebrew/")
    try:
        out = subprocess.check_output(
            ["brew", "--prefix"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return defaults
    if not out.startswith("/"):
        return defaults
    detected = out.rstrip("/") + "/"
    if detected in defaults:
        return defaults
    # Put detected prefix first so resolution prefers libs from the active
    # Homebrew install for this build.
    return (detected, *defaults)


HOMEBREW_PREFIXES = _detect_homebrew_prefixes()
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


def get_load_dylibs(p: Path) -> list[str]:
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


def _resolve_homebrew_basename(name: str) -> Path | None:
    """Find a dylib by basename in any Homebrew lib dir (cached glob)."""
    if SKIP_BASENAME_RE.match(name):
        return None
    cached = _resolve_homebrew_basename._cache  # type: ignore[attr-defined]
    if name in cached:
        return cached[name]
    for pref in HOMEBREW_PREFIXES:
        for sub in ("lib", "opt/*/lib", "Cellar/*/*/lib"):
            for cand in Path(pref).glob(f"{sub}/{name}"):
                if not cand.exists():
                    continue
                try:
                    result = cand.resolve()
                except OSError:
                    continue
                cached[name] = result
                return result
    cached[name] = None
    return None


_resolve_homebrew_basename._cache = {}  # type: ignore[attr-defined]


def bundle(framework_dir: Path) -> None:
    bin_dir = framework_dir / "bin"
    lib_dir = framework_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)

    seeds = collect_machos(bin_dir) + collect_machos(lib_dir)
    log(f"seed mach-o files: {len(seeds)}")

    # Pre-index Mach-O basenames already present in the framework's lib tree
    # (MeshLib's own libs + thirdparty-built libs copied in by
    # distribution_apple.sh) so we don't re-bundle them from Homebrew.
    own_libnames: set[str] = {p.name for p in collect_machos(lib_dir)}

    # BFS: copy every Homebrew dep transitively into lib_dir.
    queue: list[Path] = list(seeds)
    bundled: dict[str, Path] = {}        # basename -> dst path
    source_of: dict[Path, Path] = {}     # dst path -> original source dir
    visited: set[Path] = set()

    def bundle_from(src: Path, name: str | None = None) -> Path | None:
        # `name` is the basename the *referrer* asks for (e.g. the load
        # command's @rpath/<name> or the absolute path's basename). We
        # preserve it so dyld's name lookup matches even when the real file
        # behind a SOMAJOR symlink has a different version-tagged name
        # (e.g. libglfw.3.dylib -> libglfw.3.4.dylib in the bottle).
        if name is None:
            name = src.name
        if name in bundled:
            return bundled[name]
        if not src.exists():
            return None
        dst = lib_dir / name
        log(f"copy {src} -> {dst}")
        shutil.copy2(src, dst, follow_symlinks=True)
        make_writable(dst)
        bundled[name] = dst
        source_of[dst] = src.parent
        return dst

    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        cur_src_dir = source_of.get(cur)
        for dep in get_load_dylibs(cur):
            target: Path | None = None
            if should_bundle(dep):
                req_name = Path(dep).name
                src = Path(dep)
                try:
                    src = src.resolve()
                except OSError:
                    pass
                if not src.exists():
                    # Original link-time path is gone from the current bottle
                    # (the very drift we're guarding against). Fall back to
                    # the basename under the standard Homebrew lib dir.
                    fallback = _resolve_homebrew_basename(req_name)
                    if fallback is None:
                        log(f"WARN: cannot resolve {dep}; skipping")
                        continue
                    src = fallback
                target = bundle_from(src, req_name)
            elif dep.startswith(("@rpath/", "@loader_path/")):
                # Modern Homebrew bottles install with @rpath/<basename>, so
                # any MeshLib binary linked against them references the dep
                # by @rpath too -- not by an absolute /usr/local/... path. We
                # still need to bundle it. Skip libs MeshLib already ships
                # (its own dylibs / thirdparty libs put in lib/ by
                # distribution_apple.sh).
                name = dep.split("/", 1)[1]
                if name in own_libnames or name in bundled:
                    pass  # nothing to do; MeshLib provides it
                else:
                    # Inside a Homebrew bottle we already bundled, sibling
                    # deps live next to the source (e.g. gdcm's internal
                    # @rpath/libsocketxx.1.2.dylib at @loader_path/.).
                    if cur_src_dir is not None:
                        cand = cur_src_dir / name
                        if cand.exists():
                            target = bundle_from(cand.resolve(), name)
                    if target is None:
                        fb = _resolve_homebrew_basename(name)
                        if fb is not None:
                            target = bundle_from(fb, name)
            if target is not None and target not in visited:
                queue.append(target)

    # Rewrite ids, load commands, and add rpaths on every Mach-O we ship
    all_files = collect_machos(bin_dir) + collect_machos(lib_dir)
    for p in all_files:
        make_writable(p)
        is_dylib = p.suffix == ".dylib" or p.is_relative_to(lib_dir)
        if is_dylib:
            install_name_tool("-id", f"@rpath/{p.name}", str(p))

        for dep in get_load_dylibs(p):
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
