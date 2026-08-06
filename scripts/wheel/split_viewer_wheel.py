"""
Produces the optional `meshlib-viewer` wheel (mrviewerpy + the libraries only it
needs + the large CJK UI font) as the complement of two wheel-repair runs, letting
auditwheel/delvewheel/delocate decide which libraries belong where:

1. the wheel is built once with all modules;
2. a copy WITHOUT `mrviewerpy` and the CJK font (`make_base_input`) is repaired
   alongside the full wheel;
3. the repaired base-only wheel ships as `meshlib`, and everything the full repair
   contains on top of it becomes `meshlib-viewer` (`extract_viewer_wheel`).

The repair tools derive mangled library names from file contents, so the two repair
runs name their common libraries identically and the diff is exact.

Fully independent wheels would not work here: a viewer wheel repaired on its own
would reference `libMRMesh.so` while the base wheel ships `libMRMesh-<hash>.so`.
For the same reason the two wheels are only compatible in exactly matching versions:
the viewer wheel pins `meshlib==<version>` and is published for every release.

Both wheels install into the same site-packages directories (`meshlib/`,
`meshlib.libs/`, `meshlib/.dylibs/`), so the base wheel's rpaths / DLL directory /
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


def extract_viewer_wheel(full_repaired, base_repaired):
    """Write the meshlib-viewer wheel (next to the repaired base wheel) from the
    files that the full repair produced and the base repair did not."""
    full_repaired, base_repaired = Path(full_repaired), Path(base_repaired)
    name, version, rest = base_repaired.name.split("-", 2)
    assert name == "meshlib", base_repaired
    viewer_path = base_repaired.with_name(f"meshlib_viewer-{version}-{rest}")

    with zipfile.ZipFile(full_repaired) as full, zipfile.ZipFile(base_repaired) as base:
        def payload(names):
            return { n for n in names if ".dist-info/" not in n }
        base_names = payload(base.namelist())
        full_names = payload(full.namelist())
        # common libraries must have identical mangled names in both repair runs
        assert base_names <= full_names, f"repair runs diverged: {sorted(base_names - full_names)}"
        viewer_names = full_names - base_names
        assert any("mrviewerpy" in n for n in viewer_names) and any("MRViewer" in n for n in viewer_names), \
            f"unexpected viewer file set: {sorted(viewer_names)}"

        metadata = f"""\
Metadata-Version: 2.4
Name: meshlib-viewer
Version: {version}
Summary: MeshLib viewer module (mrviewerpy) with its native dependencies and UI fonts
Requires-Python: >=3.8
Requires-Dist: meshlib=={version}
Project-URL: Homepage, https://meshlib.io/
"""
        dist_info = f"meshlib_viewer-{version}.dist-info"
        with zipfile.ZipFile(viewer_path, "w", zipfile.ZIP_DEFLATED) as out:
            rows = []
            for info in full.infolist():
                if info.filename not in viewer_names:
                    continue
                data = full.read(info)
                out.writestr(info, data)
                rows.append(_record_entry(info.filename, data))
            wheel_meta = full.read(next(n for n in full.namelist() if n.endswith(".dist-info/WHEEL")))
            for name_, data in ((f"{dist_info}/METADATA", metadata.encode()), (f"{dist_info}/WHEEL", wheel_meta)):
                out.writestr(name_, data)
                rows.append(_record_entry(name_, data))
            record = io.StringIO()
            writer = csv.writer(record, lineterminator="\n")
            writer.writerows(rows)
            writer.writerow([f"{dist_info}/RECORD", "", ""])
            out.writestr(f"{dist_info}/RECORD", record.getvalue())

    print(f"Viewer wheel extracted ({len(viewer_names)} files):")
    print(f"  base:   {base_repaired} ({base_repaired.stat().st_size / 2**20:.1f} MB)")
    print(f"  viewer: {viewer_path} ({viewer_path.stat().st_size / 2**20:.1f} MB)")
