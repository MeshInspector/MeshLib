#!/usr/bin/env python3
"""Verify the bundled third-party license notices are complete and current.

The `thirdparty/licenses/` folder is hand-curated: it holds the verbatim upstream
LICENSE/NOTICE texts of every OSS component shipped in the MeshLib SDK. This script is
the drift tripwire for that folder. For each component in `manifest.json` it:

  1. checks the declared license file(s) exist and are non-empty;
  2. recomputes the component's current version from its source (git submodule SHA,
     vcpkg overlay-port version, vcpkg baseline, or a hash of tracked in-tree files)
     and fails if it differs from the pinned `version` in the manifest -- forcing a
     human to re-check the upstream license text and re-pin;
  3. checks there are no orphan license directories, and warns about git submodules
     that look shippable but are absent from the manifest.

Run `--update-versions` to re-pin the manifest to current versions after you have
verified the texts are still correct.

Invoked with no build artifacts (pre-commit / CI): every version signal is read from
the git tree and in-tree files, so it works on a fresh checkout without submodules.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LICENSES_DIR = REPO_ROOT / "thirdparty" / "licenses"
REL = LICENSES_DIR.relative_to(REPO_ROOT).as_posix()  # "thirdparty/licenses"
MANIFEST = LICENSES_DIR / "manifest.json"
VCPKG_JSON = REPO_ROOT / "thirdparty" / "vcpkg" / "vcpkg.json"
VCPKG_PORTS = REPO_ROOT / "thirdparty" / "vcpkg" / "ports"

# Directory entries in thirdparty/licenses/ that are not component folders.
NON_COMPONENT_ENTRIES = {"manifest.json", "README.md"}

# Submodules that ship nothing in the SDK binaries, so they need no license folder.
# Keep this list short and justified -- anything not here must appear in the manifest.
EXCLUDED_SUBMODULES = {
    "thirdparty/googletest",          # unit-test framework, not shipped
    "thirdparty/mrbind",              # build-time binding generator, not shipped
    "thirdparty/openvdb/v9/openvdb",  # legacy OpenVDB; only v10 ships (see OpenVDB entry)
    "test_data",                      # test assets
}


def git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def submodule_sha(path):
    """Pinned gitlink SHA of a submodule, read from the committed tree (no checkout needed)."""
    out = git("ls-tree", "HEAD", path).strip()
    if not out:
        raise ValueError(f"no tree entry for '{path}'")
    mode, kind, sha = out.split()[:3]
    if kind != "commit":
        raise ValueError(f"'{path}' is not a submodule gitlink (found {kind})")
    return sha


def vcpkg_baseline():
    data = json.loads(VCPKG_JSON.read_text(encoding="utf-8"))
    return data["configuration"]["default-registry"]["baseline"]


def vcpkg_overlay_version(port):
    data = json.loads((VCPKG_PORTS / port / "vcpkg.json").read_text(encoding="utf-8"))
    ver = next((data[k] for k in
                ("version", "version-semver", "version-string", "version-date")
                if k in data), None)
    if ver is None:
        raise ValueError(f"overlay port '{port}' has no version field")
    return f"{ver}#{data.get('port-version', 0)}"


def files_hash(track):
    """Short hash over the git blob OIDs of the tracked files, in listed order.

    Uses the OIDs git records in the tree (not working-tree bytes), so the value is
    identical on every platform regardless of line-ending normalization or autocrlf.
    """
    oids = []
    for rel in track:
        try:
            oids.append(git("rev-parse", f"HEAD:{rel}").strip())
        except subprocess.CalledProcessError:
            raise ValueError(f"tracked file not in git tree: {rel}")
    return hashlib.sha256(",".join(oids).encode()).hexdigest()[:16]


def current_version(source):
    """Recompute a component's version from its source, or None for manual entries."""
    t = source["type"]
    if t == "submodule":
        return submodule_sha(source["path"])
    if t == "vcpkg-overlay":
        return vcpkg_overlay_version(source["port"])
    if t == "vcpkg-registry":
        return vcpkg_baseline()
    if t == "hash":
        return files_hash(source["track"])
    if t == "manual":
        return None
    raise ValueError(f"unknown source type '{t}'")


def load_components():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data, data["components"]


def check():
    _, components = load_components()
    errors, warnings = [], []
    seen_ids, seen_dirs = set(), set()

    for comp in components:
        cid = comp["id"]
        if cid in seen_ids:
            errors.append(f"{cid}: duplicate manifest entry")
            continue
        seen_ids.add(cid)
        seen_dirs.add(cid)

        comp_dir = LICENSES_DIR / cid
        if not comp_dir.is_dir():
            errors.append(f"{cid}: missing license directory {REL}/{cid}/")
            continue

        # 1. declared files present and non-empty
        for rel in comp["files"]:
            f = comp_dir / rel
            if not f.is_file():
                errors.append(f"{cid}: missing license file {rel}")
            elif f.stat().st_size == 0:
                errors.append(f"{cid}: empty license file {rel}")

        # 2. version tripwire
        try:
            cur = current_version(comp["source"])
        except (ValueError, subprocess.CalledProcessError) as e:
            errors.append(f"{cid}: cannot compute version ({e})")
            continue
        if cur is None:
            continue  # manual: presence-checked only
        pinned = comp.get("version", "")
        if not pinned:
            errors.append(f"{cid}: version not pinned -- run "
                          f"'scripts/check_third_party_licenses.py --update-versions'")
        elif pinned != cur:
            errors.append(f"{cid}: version changed {pinned} -> {cur} -- re-verify the "
                          f"upstream LICENSE text, update {REL}/{cid}/, "
                          f"then re-pin with --update-versions")

    # 3a. orphan directories with no manifest entry
    for entry in sorted(LICENSES_DIR.iterdir()):
        if entry.name in NON_COMPONENT_ENTRIES:
            continue
        if entry.is_dir() and entry.name not in seen_dirs:
            errors.append(f"{entry.name}: license directory has no manifest.json entry")

    # 3b. shippable-looking submodules missing from the manifest (warning only)
    covered = {c["source"].get("path") for c in components
               if c["source"]["type"] == "submodule"}
    for path in gitmodules_paths():
        if path in covered or path in EXCLUDED_SUBMODULES:
            continue
        warnings.append(f"submodule '{path}' is not in manifest.json and not in "
                        f"EXCLUDED_SUBMODULES -- add a license entry or exclude it")

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} problem(s) found in third-party license notices.",
              file=sys.stderr)
        return False
    print(f"third-party license notices OK: {len(components)} components verified.")
    return True


def gitmodules_paths():
    gm = REPO_ROOT / ".gitmodules"
    if not gm.is_file():
        return []
    out = git("config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$")
    return [line.split(maxsplit=1)[1] for line in out.splitlines() if line.strip()]


def update_versions():
    data, components = load_components()
    changed = 0
    for comp in components:
        try:
            cur = current_version(comp["source"])
        except (ValueError, subprocess.CalledProcessError) as e:
            print(f"ERROR: {comp['id']}: cannot compute version ({e})", file=sys.stderr)
            return False
        if cur is not None and comp.get("version", "") != cur:
            print(f"  {comp['id']}: {comp.get('version', '') or '(unpinned)'} -> {cur}")
            comp["version"] = cur
            changed += 1
    # Write explicit LF bytes: Path.write_text would translate to CRLF on Windows,
    # producing spurious end-of-line churn when a Windows dev re-pins.
    MANIFEST.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"Re-pinned {changed} version(s) in {MANIFEST.relative_to(REPO_ROOT)}.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-versions", action="store_true",
                        help="re-pin manifest versions to current (after verifying texts)")
    ns = parser.parse_args()
    ok = update_versions() if ns.update_versions else check()
    sys.exit(0 if ok else 1)
