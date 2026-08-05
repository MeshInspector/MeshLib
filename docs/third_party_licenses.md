# Third-party license notices

[`thirdparty/licenses/THIRD-PARTY-NOTICES.txt`](../thirdparty/licenses/THIRD-PARTY-NOTICES.txt)
holds the verbatim upstream `LICENSE` / `NOTICE` texts of every open-source component bundled
in the MeshLib SDK, and ships **as-is** in every distribution (deb, macOS pkg, Windows folder,
Python wheel, NuGet), satisfying each upstream license's obligation to accompany the binaries.
It is hand-maintained -- there is no generation step -- and it is **separate from and additional
to** MeshLib's own top-level `LICENSE`, which covers only MeshLib itself.

## Layout

- One section per component, in `manifest.json` order. A section starts with a 4-line block --
  an 80-char `#` rule, `<id> -- <license>`, the upstream URL, another `#` rule -- followed by
  the license text(s). Components with several texts (e.g. c-blosc's per-part licenses,
  FreeType's dual license) separate them with `----- <name> -----` sub-headers, and a section
  may include a note clarifying how MeshLib uses the component (e.g. OpenCASCADE's dynamic-only
  linking).
- `thirdparty/licenses/manifest.json` is the structured index: for each component it records
  the modules that bundle it, the SPDX-ish license id, the upstream, and the **version the
  text was curated against**.

The inclusion list is `doxygen/general_pages/ThirdpartyList.dox`, reconciled against
`.gitmodules` and `thirdparty/vcpkg/vcpkg.json`. Build- and test-only submodules (googletest,
mrbind) are not shipped and are excluded (see `EXCLUDED_SUBMODULES` in the checker).

## Shipping

Every channel copies the committed file unchanged:

- **deb / macOS / vcpkg** -- `install(FILES ...)` in the top-level CMakeLists.txt, to
  `${MR_RESOURCES_DIR}`.
- **Windows folder** -- `make_install_folder.py` copies it to the install root.
- **Python wheel** -- `build_wheel.py` copies it to the wheel root; `pyproject.toml` lists it
  in `license-files`, so it lands in the wheel's `.dist-info/licenses/` (PEP 639).
- **NuGet** -- `generate_nuget_spec.py` copies it; the nuspec ships it at the package root.

## Why hand-maintained

The texts cannot be harvested reliably from a clean checkout: submodule licenses are in-tree,
but vcpkg-sourced ones (Boost, OpenCASCADE, FreeType, ...) only appear after a build, and some
(fonts, Python, CUDA) ship no machine-readable license at all. So the file is maintained by
hand -- and guarded by a drift tripwire.

## Maintenance contract (the tripwire)

`scripts/check_third_party_licenses.py` verifies every manifest component has a matching
non-empty section and that each dependency's version has **not moved** since its text was
curated. It runs **on every push and pull request, daily, and on release**
(`.github/workflows/check-third-party-licenses.yml`): the check needs no build and no
submodule checkout, so gating a PR costs only a runner slot and a few seconds, and catching
a bump on the PR that makes it beats rediscovering it at release. The daily run still covers
drift that lands without a version change. Run it locally any time:
`python scripts/check_third_party_licenses.py`.

When it reports drift:

1. `"<id>: version changed A -> B"`.
2. Re-check the upstream license for the new version; if it changed, update the component's
   section in `THIRD-PARTY-NOTICES.txt` (and the manifest `license` field if needed).
3. Re-pin: `python scripts/check_third_party_licenses.py --update-versions`, and commit the
   updated `manifest.json`.

Version is tracked per source (see `manifest.json` `_comment`): git submodule SHA, vcpkg
overlay-port version, the baseline version of each registry port the component covers (sound
because `vcpkg.json` has no per-port overrides), or a sha256 of tracked in-tree files (vendored
code, fonts, Python zips).

Registry ports are tracked **per port**, not by the registry baseline commit: that commit moves
on every routine vcpkg bump, so using it flagged all nine vcpkg components at once even when
their own libraries were untouched -- and blanket re-pinning that noise is how a genuine license
change gets waved through. Reading the ports' own baseline versions is the one signal that needs
the network (a single `versions/baseline.json` fetch from the registry, shared across components).

## Adding a new dependency

Append its section to `THIRD-PARTY-NOTICES.txt` (keeping manifest order), add an entry to
`manifest.json` (pick the `source.type` that matches how it enters the build), run
`--update-versions` to pin it, and confirm a green `python scripts/check_third_party_licenses.py`.
A shippable submodule that is neither in the manifest nor in `EXCLUDED_SUBMODULES` makes the
checker warn.
