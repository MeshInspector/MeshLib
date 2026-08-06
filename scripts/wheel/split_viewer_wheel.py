"""
Splits a repaired meshlib wheel into the slim base wheel and a `meshlib-viewer` wheel.

The viewer wheel contains `mrviewerpy` and every bundled native library that only
`mrviewerpy` needs (the viewer/UI stack: MRViewer, MRMcp, imgui, glfw, networking,
input-device libs, ...). The set is computed from the binaries' import tables, so it
follows dependency changes automatically. Like meshlib-fonts, the viewer wheel installs
its files into the same site-packages directories as the base wheel (`meshlib/`,
`meshlib.libs/`, `meshlib/.dylibs/`), where the base wheel's rpaths/DLL directory
already point.

The split runs AFTER auditwheel/delvewheel/delocate: the repair tools mangle bundled
library names per build, so the two wheels are only compatible in exactly matching
versions - the viewer wheel pins `meshlib==<version>` and is published for every release.
"""

import base64
import csv
import hashlib
import io
import struct
import zipfile
from pathlib import Path


# --- minimal per-format extraction of "which shared libraries does this binary import" ---

def elf_soname_and_needed(data):
    assert data[:4] == b"\x7fELF"
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    soname, needed = None, []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        typ, = struct.unpack_from("<I", data, off + 4)
        if typ != 6:  # SHT_DYNAMIC
            continue
        link, = struct.unpack_from("<I", data, off + 0x28)
        sh_offset, sh_size = struct.unpack_from("<QQ", data, off + 0x18)
        loff = e_shoff + link * e_shentsize
        do, ds = struct.unpack_from("<QQ", data, loff + 0x18)
        dynstr = data[do:do + ds]
        for pos in range(sh_offset, sh_offset + sh_size, 16):
            tag, val = struct.unpack_from("<qQ", data, pos)
            if tag == 0:
                break
            if tag in (1, 14):  # DT_NEEDED, DT_SONAME
                name = dynstr[val:dynstr.index(b"\0", val)].decode()
                if tag == 1:
                    needed.append(name)
                else:
                    soname = name
    return soname, needed


def pe_imported_dlls(data):
    assert data[:2] == b"MZ"
    pe_off, = struct.unpack_from("<I", data, 0x3C)
    assert data[pe_off:pe_off + 4] == b"PE\0\0"
    nsections, = struct.unpack_from("<H", data, pe_off + 6)
    opt_size, = struct.unpack_from("<H", data, pe_off + 20)
    opt_off = pe_off + 24
    magic, = struct.unpack_from("<H", data, opt_off)
    assert magic == 0x20B, "expected PE32+"
    ddir_off = opt_off + 112
    sections = []
    sec_off = opt_off + opt_size
    for i in range(nsections):
        so = sec_off + i * 40
        va, raw_size, raw_ptr = struct.unpack_from("<III", data, so + 12)
        vsize, = struct.unpack_from("<I", data, so + 8)
        sections.append((va, max(vsize, raw_size), raw_ptr))

    def rva2off(rva):
        for va, size, raw in sections:
            if va <= rva < va + size:
                return raw + rva - va
        raise ValueError(f"rva {rva:#x} not in any section")

    def read_cstr(off):
        return data[off:data.index(b"\0", off)].decode()

    dlls = []
    # regular imports (directory 1): name RVA at descriptor offset 12, stride 20
    # delay-load imports (directory 13): name RVA at descriptor offset 4, stride 32
    for ddir_index, name_field_off, stride in ((1, 12, 20), (13, 4, 32)):
        rva, size = struct.unpack_from("<II", data, ddir_off + ddir_index * 8)
        if not rva:
            continue
        off = rva2off(rva)
        while True:
            name_rva, = struct.unpack_from("<I", data, off + name_field_off)
            if not name_rva:
                break
            dlls.append(read_cstr(rva2off(name_rva)).lower())
            off += stride
    return dlls


def macho_imported_dylibs(data):
    assert data[:4] == b"\xcf\xfa\xed\xfe", "expected thin 64-bit Mach-O"
    ncmds, = struct.unpack_from("<I", data, 16)
    off, deps = 32, []
    LOAD_DYLIB_CMDS = (0xC, 0x80000018, 0x8000001F)  # LC_LOAD_DYLIB, _WEAK_, LC_REEXPORT_DYLIB
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmd in LOAD_DYLIB_CMDS:
            name_off, = struct.unpack_from("<I", data, off + 8)
            path = data[off + name_off:data.index(b"\0", off + name_off)].decode()
            deps.append(path.rsplit("/", 1)[-1])
        off += cmdsize
    return deps


# --- wheel splitting ---

def record_entry(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [name, f"sha256={digest}", str(len(data))]


def write_record(wheel, dist_info, rows):
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    writer.writerows(rows)
    writer.writerow([f"{dist_info}/RECORD", "", ""])
    wheel.writestr(f"{dist_info}/RECORD", record.getvalue())


def viewer_only_libs(entries):
    """entries: {zip name: bytes} for all native binaries. Returns viewer-only zip names."""
    nodes = {}  # identity (soname / dll name / dylib basename) -> (zip name, deps)
    base_roots, viewer_roots = [], []
    for name, data in entries.items():
        basename = name.rsplit("/", 1)[-1]
        if data[:4] == b"\x7fELF":
            soname, deps = elf_soname_and_needed(data)
            identity = soname or basename
        elif data[:2] == b"MZ":
            identity, deps = basename.lower(), pe_imported_dlls(data)
        else:
            identity, deps = basename, macho_imported_dylibs(data)
        is_bundled_lib = name.startswith("meshlib.libs/") or "/.dylibs/" in name
        if is_bundled_lib:
            nodes[identity] = (name, deps)
        elif basename.startswith("mrviewerpy."):
            viewer_roots += deps
        else:
            base_roots += deps

    def closure(roots):
        seen, stack = set(), list(roots)
        while stack:
            n = stack.pop()
            if n in seen or n not in nodes:
                continue
            seen.add(n)
            stack += nodes[n][1]
        return seen

    base = closure(base_roots)
    return { nodes[n][0] for n in closure(viewer_roots) - base }


def split_wheel(wheel_path):
    wheel_path = Path(wheel_path)
    name, version, rest = wheel_path.name.split("-", 2)
    assert name == "meshlib", wheel_path
    viewer_path = wheel_path.with_name(f"meshlib_viewer-{version}-{rest}")

    src = zipfile.ZipFile(wheel_path)
    binaries = {}
    for info in src.infolist():
        if not (info.filename.rsplit(".", 1)[-1] in ("so", "pyd", "dll")
                or ".so." in info.filename or ".dylib" in info.filename):
            continue
        data = src.read(info)
        if data[:4] in (b"\x7fELF", b"\xcf\xfa\xed\xfe") or data[:2] == b"MZ":
            binaries[info.filename] = data
    moved = viewer_only_libs(binaries)
    moved |= { n for n in src.namelist() if n.rsplit("/", 1)[-1].startswith("mrviewerpy.") }
    base_dist_info = next(n.split("/")[0] for n in src.namelist() if n.endswith(".dist-info/RECORD"))
    wheel_meta = src.read(f"{base_dist_info}/WHEEL")

    # base wheel: everything else; drop moved names from RECORD and delvewheel's .load-order
    base_tmp = wheel_path.with_suffix(".tmp")
    moved_basenames = { n.rsplit("/", 1)[-1] for n in moved }
    with zipfile.ZipFile(base_tmp, "w", zipfile.ZIP_DEFLATED) as out:
        rows = []
        for info in src.infolist():
            if info.filename in moved:
                continue
            data = src.read(info)
            if info.filename.endswith("/RECORD"):
                continue  # rewritten below
            if info.filename.rsplit("/", 1)[-1].startswith(".load-order"):
                data = b"".join(
                    line for line in data.splitlines(keepends=True)
                    if line.strip().decode() not in moved_basenames
                )
            out.writestr(info, data)
            rows.append(record_entry(info.filename, data))
        write_record(out, base_dist_info, rows)

    # viewer wheel: the moved files + its own dist-info, pinning the exact base version
    metadata = f"""\
Metadata-Version: 2.4
Name: meshlib-viewer
Version: {version}
Summary: MeshLib viewer module (mrviewerpy) with its native dependencies
Requires-Python: >=3.8
Requires-Dist: meshlib=={version}
Project-URL: Homepage, https://meshlib.io/
"""
    dist_info = f"meshlib_viewer-{version}.dist-info"
    with zipfile.ZipFile(viewer_path, "w", zipfile.ZIP_DEFLATED) as out:
        rows = []
        for info in src.infolist():
            if info.filename not in moved:
                continue
            data = src.read(info)
            out.writestr(info, data)
            rows.append(record_entry(info.filename, data))
        for name_, data in ((f"{dist_info}/METADATA", metadata.encode()), (f"{dist_info}/WHEEL", wheel_meta)):
            out.writestr(name_, data)
            rows.append(record_entry(name_, data))
        write_record(out, dist_info, rows)

    src.close()
    base_tmp.replace(wheel_path)
    print(f"Split {len(moved)} viewer files out of {wheel_path.name}:")
    print(f"  base:   {wheel_path} ({wheel_path.stat().st_size / 2**20:.1f} MB)")
    print(f"  viewer: {viewer_path} ({viewer_path.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    import sys
    split_wheel(sys.argv[1])
