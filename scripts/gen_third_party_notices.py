#!/usr/bin/env python3
"""Concatenate the curated per-component notices in thirdparty/licenses/ into a
single THIRD-PARTY-NOTICES.txt for shipping in the SDK packages.

The per-component folder (thirdparty/licenses/<id>/ + manifest.json) is the
maintained source; this produces the aggregated file that each packaging path
(deb/macOS/vcpkg, Windows folder, wheel, NuGet) ships. Generated at package
time -- it is not committed. stdlib-only.

Usage: gen_third_party_notices.py --output <path>
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LICENSES_DIR = REPO_ROOT / "thirdparty" / "licenses"
MANIFEST = LICENSES_DIR / "manifest.json"

RULE = "#" * 80


def build():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = [
        "MeshLib third-party license notices",
        "===================================",
        "",
        "This file aggregates the upstream license texts of the third-party",
        "open-source components distributed with the MeshLib SDK. It is separate",
        "from and additional to MeshLib's own LICENSE.",
        "",
    ]
    for comp in data["components"]:
        cid = comp["id"]
        header = f"{cid} -- {comp.get('license', '')}"
        out += ["", RULE, header, comp.get("upstream", ""), RULE, ""]
        files = comp["files"]
        multi = len(files) > 1
        for rel in files:
            path = LICENSES_DIR / cid / rel
            text = path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
            if multi:
                out += [f"----- {rel} -----", ""]
            out += [text, ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="path to write THIRD-PARTY-NOTICES.txt")
    ns = parser.parse_args()
    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # explicit LF, no platform newline translation
    out_path.write_bytes(build().encode("utf-8"))
    print(f"wrote {out_path}")
