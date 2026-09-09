"""
Splits the built distribution into two wheels:

- `meshlib-core`: the headless core, i.e. the repaired viewer-less wheel, produced
  entirely by auditwheel/delvewheel/delocate;
- `meshlib`: mrviewerpy, the libraries only it needs and the CJK UI font, pinning
  `meshlib-core==<version>`.

The library sets are the file-name complement of two repair runs: the repair tools
derive mangled library names from file contents, so shared libraries get identical
names in both runs. Independently repaired wheels would not work (the viewer's libs
would reference `libMRMesh.so` while the core ships `libMRMesh-<hash>.so`), which is
also why the version pin is exact. Both wheels install into the same site-packages
directories, so the core's rpaths / DLL directory / `@loader_path` references
resolve the viewer's libraries without extra wiring.
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


def make_core_input(full_wheel, out_dir):
    """Copy of the built (not yet repaired) wheel without the viewer files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    core_wheel = out_dir / Path(full_wheel).name
    with zipfile.ZipFile(full_wheel) as src, zipfile.ZipFile(core_wheel, "w", zipfile.ZIP_DEFLATED) as out:
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
    return core_wheel


def _record_entry(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [name, f"sha256={digest}", str(len(data))]


def make_meshlib_metadata(core_metadata, version):
    """The `meshlib` METADATA is the core's setuptools-generated one (readme,
    classifiers, license refs) renamed, with the core pin replacing direct deps."""
    out = []
    dep_lines_replaced = 0
    for line in core_metadata.decode().splitlines(keepends=True):
        if line.startswith("Name: "):
            out.append("Name: meshlib\n")
        elif line.startswith("Requires-Dist: "):
            # the core's own deps come transitively; its dep lines collapse into the pin
            if dep_lines_replaced == 0:
                out.append(f"Requires-Dist: meshlib-core=={version}\n")
            dep_lines_replaced += 1
        else:
            out.append(line)
    assert dep_lines_replaced > 0, "no Requires-Dist in the core METADATA"
    return "".join(out).encode()


def extract_meshlib_wheel(full_repaired, core_repaired):
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
