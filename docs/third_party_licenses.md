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
(`.github/workflows/check-third-party-licenses.yml`): checking needs no build, no submodule
checkout and no network, so gating a PR costs only a runner slot and a few seconds, and
catching a bump on the PR that makes it beats rediscovering it at release. The daily run
still covers drift that lands without a version change. Run it locally any time:
`python scripts/check_third_party_licenses.py`.

Version is tracked per source (see `manifest.json` `_comment`): git submodule SHA, vcpkg
overlay-port version, the baseline version of each registry port the component covers (sound
because `vcpkg.json` has no per-port overrides), or a sha256 of tracked in-tree files (vendored
code, fonts, Python zips).

### Fixing drift

**A component's own version moved** -- `"<id>: version changed A -> B"`:

1. Re-check the upstream license at the new version; if the text changed, update that
   component's section in `THIRD-PARTY-NOTICES.txt` (and the manifest `license` field if the
   license itself changed).
2. Re-pin: `python scripts/check_third_party_licenses.py --update-versions`.
3. Commit the updated `manifest.json` together with any notice edits.

**The vcpkg baseline moved** -- `"vcpkg registry baseline moved A -> B"`, which is what a
routine vcpkg bump produces:

1. Run `python scripts/check_third_party_licenses.py --update-versions`. This is the only
   step that reads the vcpkg registry over the network, and it prints exactly which ports
   moved -- usually none or one or two, not all nine components.
2. For each port it names, re-check that component's upstream license and update its section
   if the text changed. Ports it does not name have not moved; nothing to re-verify.
3. Commit the updated `manifest.json` (its per-port `version` diff is the review record of
   what actually changed) together with any notice edits.

Registry ports are recorded **per port**, not by the registry baseline commit, because that
commit moves on every routine vcpkg bump: tracking it directly flagged all nine vcpkg
components at once even when their own libraries were untouched, and blanket re-pinning that
noise is how a genuine license change gets waved through. `manifest.json`'s
`vcpkg_registry_baseline` records the commit those port versions were resolved at. A port's
version is a pure function of that commit, so while it still matches `vcpkg.json` no registry
port can have moved -- which is why the check stays offline and only `--update-versions`
reaches the network.

## Adding a new dependency

Append its section to `THIRD-PARTY-NOTICES.txt` (keeping manifest order), add an entry to
`manifest.json` (pick the `source.type` that matches how it enters the build), run
`--update-versions` to pin it, and confirm a green `python scripts/check_third_party_licenses.py`.
A shippable submodule that is neither in the manifest nor in `EXCLUDED_SUBMODULES` makes the
checker warn.
