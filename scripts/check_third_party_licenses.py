#!/usr/bin/env python3
"""Verify the bundled third-party license notices are complete and current.

`thirdparty/licenses/THIRD-PARTY-NOTICES.txt` is hand-maintained: it holds the verbatim
upstream LICENSE/NOTICE texts of every OSS component shipped in the MeshLib SDK, one
'#'-ruled section per component, and every SDK package ships it as-is. This script is
the drift tripwire for that file. For each component in `manifest.json` it:

  1. checks the file has a matching non-empty section (id, license, upstream), with
     no orphan or misordered sections;
  2. recomputes the component's current version from its source (git submodule SHA,
     vcpkg overlay-port version, vcpkg baseline, or a hash of tracked in-tree files)
     and fails if it differs from the pinned `version` in the manifest -- forcing a
     human to re-check the upstream license text and re-pin;
  3. warns about git submodules that look shippable but are absent from the manifest.

Run `--update-versions` to re-pin the manifest to current versions after you have
verified the texts are still correct.

Runs daily and on release (.github/workflows/check-third-party-licenses.yml); run it
locally any time with `python scripts/check_third_party_licenses.py`. Every version signal
is read from the git tree and in-tree files, so it needs no build and no submodule checkout.
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
NOTICES = LICENSES_DIR / "THIRD-PARTY-NOTICES.txt"
VCPKG_JSON = REPO_ROOT / "thirdparty" / "vcpkg" / "vcpkg.json"
VCPKG_PORTS = REPO_ROOT / "thirdparty" / "vcpkg" / "ports"

# Section separator in THIRD-PARTY-NOTICES.txt.
RULE = "#" * 80

# The only entries allowed in thirdparty/licenses/.
ALLOWED_ENTRIES = {"manifest.json", "THIRD-PARTY-NOTICES.txt"}

# Submodules that ship nothing in the SDK binaries, so they need no license folder.
# Keep this list short and justified -- anything not here must appear in the manifest.
EXCLUDED_SUBMODULES = {
    "thirdparty/googletest",          # unit-test framework, not shipped
    "thirdparty/mrbind",              # build-time binding generator, not shipped
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


def parse_sections(text):
    """Parse THIRD-PARTY-NOTICES.txt into [(id, license, upstream, has_body)].

    A section starts with a 4-line block: RULE, '<id> -- <license>', upstream, RULE;
    its body runs to the next such block (or EOF).
    """
    lines = text.split("\n")

    def is_section_start(i):
        return lines[i] == RULE and i + 3 < len(lines) and lines[i + 3] == RULE

    starts = [i for i in range(len(lines)) if is_section_start(i)]
    sections = []
    for n, i in enumerate(starts):
        header, upstream = lines[i + 1], lines[i + 2]
        cid, sep, lic = header.partition(" -- ")
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body = "\n".join(lines[i + 4:end]).strip()
        sections.append((cid if sep else header, lic, upstream, bool(body)))
    return sections


def check():
    _, components = load_components()
    errors, warnings = [], []
    seen_ids = set()

    notices_rel = f"{REL}/{NOTICES.name}"
    sections = []
    if NOTICES.is_file():
        sections = parse_sections(NOTICES.read_text(encoding="utf-8"))
    else:
        errors.append(f"missing {notices_rel}")
    by_id = {}
    for cid, lic, upstream, has_body in sections:
        if cid in by_id:
            errors.append(f"{cid}: duplicate section in {notices_rel}")
        by_id[cid] = (lic, upstream, has_body)

    for comp in components:
        cid = comp["id"]
        if cid in seen_ids:
            errors.append(f"{cid}: duplicate manifest entry")
            continue
        seen_ids.add(cid)

        # 1. matching non-empty section in THIRD-PARTY-NOTICES.txt
        sec = by_id.get(cid)
        if sec is None:
            errors.append(f"{cid}: no section in {notices_rel}")
        else:
            lic, upstream, has_body = sec
            if lic != comp.get("license", ""):
                errors.append(f"{cid}: section license '{lic}' != manifest "
                              f"'{comp.get('license', '')}'")
            if upstream != comp.get("upstream", ""):
                errors.append(f"{cid}: section upstream '{upstream}' != manifest "
                              f"'{comp.get('upstream', '')}'")
            if not has_body:
                errors.append(f"{cid}: empty section in {notices_rel}")

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
                          f"upstream LICENSE text, update its section in {notices_rel}, "
                          f"then re-pin with --update-versions")

    # 3a. orphan or misordered sections, stray files in the folder
    for cid, *_ in sections:
        if cid not in seen_ids:
            errors.append(f"{cid}: section in {notices_rel} has no manifest.json entry")
    file_order = [cid for cid, *_ in sections if cid in seen_ids]
    manifest_order = [c["id"] for c in components if c["id"] in by_id]
    if file_order != manifest_order:
        errors.append(f"sections in {notices_rel} are not in manifest.json order")
    for entry in sorted(LICENSES_DIR.iterdir()):
        if entry.name not in ALLOWED_ENTRIES:
            errors.append(f"unexpected entry {REL}/{entry.name} -- only "
                          f"{', '.join(sorted(ALLOWED_ENTRIES))} belong here")

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
