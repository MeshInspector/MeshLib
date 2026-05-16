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

Why not dylibbundler / CMake BundleUtilities (`fixup_bundle`):

Decisive reason -- load-command shape under dual load context:
  Both tools rewrite LC_LOAD_DYLIB entries to @executable_path-relative
  paths (fixup_bundle writes "@executable_path/../Frameworks/libfoo.dylib";
  dylibbundler writes "@rpath/libfoo.dylib" resolved through a fixed rpath
  that itself points at "@executable_path/../libs/"). Either form works
  for a .app where every load is initiated by a binary inside the bundle,
  so @executable_path always points at the bundle.

  MeshLib's lib/ is also loaded by external python3 (when a user does
  `import meshlib` from a regular interpreter), in which case
  @executable_path is e.g. /usr/local/bin/python3 -- and an
  "@executable_path/../lib/libfoo.dylib" load would resolve to
  /usr/local/lib/libfoo.dylib, outside the framework. To keep both load
  contexts working we need bare @rpath/<name> load commands plus *two*
  rpath entries -- @executable_path/../lib for binaries in bin/ and
  @loader_path/. for dylibs in lib/. fixup_bundle's single-embedded_path
  model doesn't express that.

Secondary points:
  - dylibbundler 1.0.5 (the version Homebrew ships) hardcodes /usr/local
    and /opt/homebrew as the only search prefixes; the arm64 self-hosted
    build runner installs Homebrew at /Users/runner/.homebrew. This
    script calls `brew --prefix` at startup. (fixup_bundle accepts DIRS
    so it could resolve that, but the load-command shape above still
    rules it out.)
  - otool / install_name_tool / codesign ship with Xcode CLT, already a
    build requirement, so no new build-time tool to install either way.
"""
from __future__ import annotations

import argparse
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

try:
    from delocate.tools import (
        add_rpath,
        get_install_names,
        get_rpaths,
        set_install_id,
        set_install_name,
    )
except ImportError:
    sys.exit(
        "scripts/macos_bundle_dylibs.py requires the 'delocate' Python "
        "package. Install with: pip install delocate==0.10.7"
    )

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


def codesign_adhoc(p: Path) -> None:
    # delocate.tools's helpers ad-hoc sign as a side effect, but they don't
    # carry over entitlements / runtime flags. After all rewrites, do one
    # final --preserve-metadata sign per file so any flags the linker
    # embedded (e.g. hardened runtime on arm64) survive.
    subprocess.check_call([
        "codesign", "--force", "--sign", "-",
        "--preserve-metadata=entitlements,requirements,flags,runtime",
        str(p),
    ])


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
        for dep in get_install_names(str(cur)):
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

    # Rewrite ids, load commands, and add rpaths on every Mach-O we ship.
    # delocate.tools.* helpers wrap install_name_tool and ad-hoc sign after
    # each call; the final codesign_adhoc preserves entitlements/flags that
    # delocate's default signing would drop.
    all_files = collect_machos(bin_dir) + collect_machos(lib_dir)
    for p in all_files:
        make_writable(p)
        sp = str(p)
        is_dylib = p.suffix == ".dylib" or p.is_relative_to(lib_dir)
        if is_dylib:
            set_install_id(sp, f"@rpath/{p.name}", ad_hoc_sign=False)

        for dep in get_install_names(sp):
            if not should_bundle(dep):
                continue
            set_install_name(
                sp, dep, f"@rpath/{Path(dep).name}", ad_hoc_sign=False,
            )

        rpath = "@executable_path/../lib" if p.is_relative_to(bin_dir) else "@loader_path/."
        if rpath not in get_rpaths(sp):
            add_rpath(sp, rpath, ad_hoc_sign=False)

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
