"""
Splits the Python distribution into two wheels:

- `meshlib-core` — the headless core (mrmeshpy, mrmeshnumpy, mrcudapy and their
  native libraries). Depends only on numpy.
- `meshlib` — the viewer add-on (mrviewerpy, the libraries only it needs and the
  large CJK UI font), which depends on `meshlib-core==<version>`. Keeping the
  `meshlib` name on the full package means `pip install meshlib` and upgrades of
  existing installations behave exactly as today, while headless deployments can
  switch to `pip install meshlib-core`.

The split is the complement of two wheel-repair runs, letting
auditwheel/delvewheel/delocate decide which libraries belong where:

1. the wheel is built once with all modules (project name `meshlib-core`);
2. a copy WITHOUT `mrviewerpy` and the CJK font (`make_base_input`) is repaired
   alongside the full wheel;
3. the repaired core-only wheel ships as `meshlib-core`, and everything the full
   repair contains on top of it becomes `meshlib` (`extract_viewer_wheel`), whose
   dist-info is the core's setuptools-generated one with the name and dependencies
   patched.

The repair tools derive mangled library names from file contents, so the two repair
runs name their common libraries identically and the diff is exact.

Fully independent wheels would not work here: a viewer wheel repaired on its own
would reference `libMRMesh.so` while the core wheel ships `libMRMesh-<hash>.so`.
For the same reason the two wheels are only compatible in exactly matching versions:
`meshlib` pins `meshlib-core==<version>` and both are published for every release.

Both wheels install into the same site-packages directories (`meshlib/`,
`meshlib.libs/`, `meshlib/.dylibs/`), so the core wheel's rpaths / DLL directory /
`@loader_path` references resolve the viewer's libraries with no extra wiring.
"""

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

# only the viewer UI renders the CJK font; label rendering treats it as optional
VIEWER_FILE_PREFIXES = ("mrviewerpy.", "NotoSansCJK-Regular.ttc")


def _is_viewer_source_file(zip_name):
    return zip_name.rsplit("/", 1)[-1].startswith(VIEWER_FILE_PREFIXES)


def make_base_input(full_wheel, out_dir):
    """Copy of the built (not yet repaired) wheel without the viewer files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    base_wheel = out_dir / Path(full_wheel).name
    with zipfile.ZipFile(full_wheel) as src, zipfile.ZipFile(base_wheel, "w", zipfile.ZIP_DEFLATED) as out:
        for info in src.infolist():
            if _is_viewer_source_file(info.filename):
                continue
            data = src.read(info)
            if info.filename.endswith("/RECORD"):
                data = b"".join(
                    line for line in data.splitlines(keepends=True)
                    if not _is_viewer_source_file(line.split(b",", 1)[0].decode())
                )
            out.writestr(info, data)
    return base_wheel


def _record_entry(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [name, f"sha256={digest}", str(len(data))]


def make_meshlib_metadata(core_metadata, version):
    """The `meshlib` METADATA is the core's setuptools-generated one (readme,
    classifiers, license refs) renamed, with the core pin replacing direct deps."""
    lines = core_metadata.decode().splitlines(keepends=True)
    out = []
    for line in lines:
        if line.startswith("Name: "):
            out.append("Name: meshlib\n")
        elif line.startswith("Requires-Dist: "):
            continue  # numpy etc. come transitively via the core
        elif line.startswith("Requires-Python: "):
            out.append(line)
            out.append(f"Requires-Dist: meshlib-core=={version}\n")
        else:
            out.append(line)
    return "".join(out).encode()


def extract_viewer_wheel(full_repaired, core_repaired):
    """Write the `meshlib` wheel (next to the repaired core wheel) from the files
    that the full repair produced and the core repair did not."""
    full_repaired, core_repaired = Path(full_repaired), Path(core_repaired)
    name, version, rest = core_repaired.name.split("-", 2)
    assert name == "meshlib_core", core_repaired
    meshlib_path = core_repaired.with_name(f"meshlib-{version}-{rest}")

    with zipfile.ZipFile(full_repaired) as full, zipfile.ZipFile(core_repaired) as core:
        def payload(names):
            return { n for n in names if ".dist-info/" not in n }
        core_names = payload(core.namelist())
        full_names = payload(full.namelist())
        # common libraries must have identical mangled names in both repair runs
        assert core_names <= full_names, f"repair runs diverged: {sorted(core_names - full_names)}"
        viewer_names = full_names - core_names
        assert any("mrviewerpy" in n for n in viewer_names) and any("MRViewer" in n for n in viewer_names), \
            f"unexpected viewer file set: {sorted(viewer_names)}"

        core_dist_info = next(n for n in core.namelist() if n.endswith(".dist-info/METADATA")).rsplit("/", 1)[0]
        dist_info = f"meshlib-{version}.dist-info"
        with zipfile.ZipFile(meshlib_path, "w", zipfile.ZIP_DEFLATED) as out:
            rows = []
            for info in full.infolist():
                if info.filename not in viewer_names:
                    continue
                data = full.read(info)
                out.writestr(info, data)
                rows.append(_record_entry(info.filename, data))
            extra_entries = [
                (f"{dist_info}/METADATA", make_meshlib_metadata(core.read(f"{core_dist_info}/METADATA"), version)),
                (f"{dist_info}/WHEEL", core.read(f"{core_dist_info}/WHEEL")),
            ]
            # license files referenced by METADATA's License-File fields
            extra_entries += [
                (f"{dist_info}/licenses/{n.rsplit('/', 1)[-1]}", core.read(n))
                for n in core.namelist()
                if n.startswith(f"{core_dist_info}/licenses/")
            ]
            for name_, data in extra_entries:
                out.writestr(name_, data)
                rows.append(_record_entry(name_, data))
            record = io.StringIO()
            writer = csv.writer(record, lineterminator="\n")
            writer.writerows(rows)
            writer.writerow([f"{dist_info}/RECORD", "", ""])
            out.writestr(f"{dist_info}/RECORD", record.getvalue())

    print(f"meshlib (viewer) wheel extracted ({len(viewer_names)} files):")
    print(f"  core:    {core_repaired} ({core_repaired.stat().st_size / 2**20:.1f} MB)")
    print(f"  meshlib: {meshlib_path} ({meshlib_path.stat().st_size / 2**20:.1f} MB)")
