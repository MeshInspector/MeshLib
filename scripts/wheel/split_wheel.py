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

Each wheel carries the platform tag of the repair run that produced its libraries:
the viewer's dependencies are what need the newer glibc, so tagging both wheels
alike would either over-promise for `meshlib` or under-promise for the core.
"""

import base64
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

# only the viewer UI renders the CJK font; label rendering treats it as optional
VIEWER_FILE_PREFIXES = ("mrviewerpy.", "NotoSansCJK-Regular.ttc")

MANYLINUX_TAG_RE = re.compile(r"manylinux_(\d+)_(\d+)_\w+")


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


def _write_record(out, dist_info, rows):
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    writer.writerows(rows)
    writer.writerow([f"{dist_info}/RECORD", "", ""])
    out.writestr(f"{dist_info}/RECORD", record.getvalue())


def _platform_tags(wheel_path):
    return Path(wheel_path).stem.split("-")[-1].split(".")


def _lowest_glibc_tag(tags):
    """The single tag a repair run's tag set collapses to: auditwheel adds every
    manylinux policy the wheel turned out to satisfy, and the oldest glibc one
    implies all the newer ones."""
    if len(tags) == 1:
        return tags[0]
    assert all(MANYLINUX_TAG_RE.fullmatch(tag) for tag in tags), f"cannot order tags: {tags}"
    return min(tags, key=lambda tag: tuple(int(n) for n in MANYLINUX_TAG_RE.fullmatch(tag).groups()))


def _narrow_wheel_tags(wheel_metadata, plat_tag):
    """The WHEEL file with its `Tag:` lines narrowed to one platform."""
    lines = [
        line for line in wheel_metadata.decode().splitlines(keepends=True)
        if not line.startswith("Tag: ") or line.rstrip().rsplit("-", 1)[-1] == plat_tag
    ]
    assert any(line.startswith("Tag: ") for line in lines), f"no {plat_tag} tag in WHEEL"
    return "".join(lines).encode()


def retag_wheel(wheel_path, plat_tag):
    """Rewrite a repaired wheel to carry `plat_tag` alone, renamed to match."""
    wheel_path = Path(wheel_path)
    if _platform_tags(wheel_path) == [plat_tag]:
        return wheel_path
    new_path = wheel_path.with_name("-".join([*wheel_path.stem.split("-")[:-1], plat_tag]) + ".whl")
    with zipfile.ZipFile(wheel_path) as src, zipfile.ZipFile(new_path, "w", zipfile.ZIP_DEFLATED) as out:
        dist_info = next(n for n in src.namelist() if n.endswith(".dist-info/WHEEL")).rsplit("/", 1)[0]
        rows = []
        for info in src.infolist():
            if info.filename == f"{dist_info}/RECORD":
                continue
            data = src.read(info)
            if info.filename == f"{dist_info}/WHEEL":
                data = _narrow_wheel_tags(data, plat_tag)
            out.writestr(info, data)
            rows.append(_record_entry(info.filename, data))
        _write_record(out, dist_info, rows)
    wheel_path.unlink()
    return new_path


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
    assert core_repaired.name.split("-")[0] == "meshlib_core", core_repaired
    # the core keeps the policy its own repair run found it eligible for
    core_repaired = retag_wheel(core_repaired, _lowest_glibc_tag(_platform_tags(core_repaired)))
    meshlib_plat = _lowest_glibc_tag(_platform_tags(full_repaired))
    _, version, *py_abi_tags, _ = core_repaired.stem.split("-")
    meshlib_path = core_repaired.with_name("-".join(["meshlib", version, *py_abi_tags, meshlib_plat]) + ".whl")

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
        full_dist_info = next(n for n in full.namelist() if n.endswith(".dist-info/WHEEL")).rsplit("/", 1)[0]
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
                (f"{dist_info}/WHEEL", _narrow_wheel_tags(full.read(f"{full_dist_info}/WHEEL"), meshlib_plat)),
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
            _write_record(out, dist_info, rows)
