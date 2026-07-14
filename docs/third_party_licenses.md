# Third-party license notices

Verbatim upstream `LICENSE` / `NOTICE` texts for every open-source component bundled in the
MeshLib SDK live in [`thirdparty/licenses/`](../thirdparty/licenses/). Redistributing those
components (in the deb, macOS pkg, Windows folder, Python wheel, or NuGet package) obliges us to
ship their license texts alongside the binaries; that folder is the bundle. It is **separate
from and additional to** MeshLib's own top-level `LICENSE`, which covers only MeshLib itself.

## Layout

- One directory per component under `thirdparty/licenses/`, named for the component (`Boost/`,
  `OpenCASCADE/`, ...), holding its license file(s). Components shared by several MeshLib modules
  appear once. A component may also carry a `NOTE.txt` clarifying how MeshLib uses it (e.g.
  FreeType's dual license, OpenCASCADE's dynamic-only linking).
- `thirdparty/licenses/manifest.json` is the source of truth: for each component it records the
  modules that bundle it, the SPDX-ish license id, the upstream, the license file names, and the
  **version the text was curated against**.

The inclusion list is `doxygen/general_pages/ThirdpartyList.dox`, reconciled against
`.gitmodules` and `thirdparty/vcpkg/vcpkg.json`. Build- and test-only submodules (googletest,
mrbind) are not shipped and are excluded (see `EXCLUDED_SUBMODULES` in the checker).

## Why hand-curated

The texts cannot be harvested reliably from a clean checkout: submodule licenses are in-tree,
but vcpkg-sourced ones (Boost, OpenCASCADE, FreeType, ...) only appear after a build, and some
(fonts, Python, CUDA) ship no machine-readable license at all. So the folder is maintained by
hand -- and guarded by a drift tripwire.

## Maintenance contract (the tripwire)

`scripts/check_third_party_licenses.py` verifies the license files exist and that each
dependency's version has **not moved** since its text was curated. It runs **daily and on
release** (`.github/workflows/check-third-party-licenses.yml`) -- deliberately not on every PR,
so a routine dependency bump merges freely and this flags within a day (and hard-fails at
release) if a notice needs updating. Run it locally any time:
`python scripts/check_third_party_licenses.py`.

When it reports drift:

1. `"<id>: version changed A -> B"`.
2. Re-check the upstream license for the new version; if it changed, update the text in
   `thirdparty/licenses/<id>/` (and the `license`/`files` fields if needed).
3. Re-pin: `python scripts/check_third_party_licenses.py --update-versions`, and commit the
   updated `manifest.json`.

Version is tracked per source (see `manifest.json` `_comment`): git submodule SHA, vcpkg
overlay-port version, the vcpkg registry baseline (sound because `vcpkg.json` has no per-port
overrides), or a sha256 of tracked in-tree files (vendored code, fonts, Python zips).

## Adding a new dependency

Add its directory + license file(s) under `thirdparty/licenses/`, add an entry to
`manifest.json` (pick the `source.type` that matches how it enters the build), run
`--update-versions` to pin it, and confirm a green `python scripts/check_third_party_licenses.py`.
A shippable submodule that is neither in the manifest nor in `EXCLUDED_SUBMODULES` makes the
checker warn.
