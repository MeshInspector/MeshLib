#!/usr/bin/env python3
"""Remove the libz-ng.so.2 entry from .gnu.version_r in consumer .so files.

Why
---
Upstream zlib-ng's CMakeLists.txt enables HAVE_SYMVER and a GNU symbol
version script (zlib-ng.map) on non-Apple, non-AIX Unix, tagging every
exported symbol in libz-ng.so.2 with ZLIB_NG_2.0.0 / ZLIB_NG_2.1.0
nodes. These propagate into the DT_VERNEED of every consumer .so.

auditwheel's manylinux policy database has no entry for libz-ng.so.2,
so it rejects any wheel carrying these tags as "too-recent versioned
symbols" -- a generic phrasing for any (lib, version-tag) pair it
cannot place in a known policy.

MeshLib does not exercise zlib-ng's ABI versioning -- consumers are
rebuilt against whatever libz-ng we ship -- so it is safe to drop the
version requirement on the consumer side.

What this does
--------------
For each .so given (file paths, or directories searched recursively),
locate the Verneed entry for libz-ng.so.2 in .gnu.version_r and:

  1. Rewrite the .gnu.version_r section in place, omitting that entry.
     Remaining Verneed/Vernaux records are repacked contiguously and
     the section is zero-padded to its original size, so all other
     section offsets stay valid. (We can't just zero vn_cnt -- pyelftools
     hard-asserts vn_cnt > 0 when iterating, so auditwheel would crash.)
  2. Decrement DT_VERNEEDNUM in the dynamic table.
  3. Rewrite every .gnu.version (DT_VERSYM) entry that referenced one
     of the dropped Vernaux indices to VER_NDX_GLOBAL (1, "global,
     unversioned"), so the dynamic linker resolves zng_* symbols by
     name only at load time.

Other versioned dependencies (GLIBC_*, GLIBCXX_*, libgcc_s, ...) are
left untouched.

Run on every consumer .so before auditwheel.
"""

import io
import struct
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile
from elftools.elf.gnuversions import GNUVerNeedSection


_TARGET_LIB = "libz-ng.so.2"

# Verneed/Vernaux record sizes are identical for ELF32 and ELF64.
_VN_HDR_SIZE = 16  # vn_version(2) vn_cnt(2) vn_file(4) vn_aux(4) vn_next(4)
_VNA_SIZE = 16     # vna_hash(4) vna_flags(2) vna_other(2) vna_name(4) vna_next(4)

_VER_NDX_GLOBAL = 1
_VERSYM_HIDDEN_BIT = 0x8000

# DT_VERNEEDNUM = count of Verneed entries in the .gnu.version_r linked list.
_DT_VERNEEDNUM = 0x6fffffff


def _strip_one(so_path: Path) -> bool:
    with open(so_path, "rb") as f:
        data = bytearray(f.read())

    elf = ELFFile(io.BytesIO(bytes(data)))
    endian = "<" if elf.little_endian else ">"

    verneed_sec = next(
        (s for s in elf.iter_sections() if isinstance(s, GNUVerNeedSection)),
        None,
    )
    if verneed_sec is None:
        return False
    dynstr = elf.get_section_by_name(".dynstr")
    if dynstr is None:
        return False

    section_offset = verneed_sec["sh_offset"]
    section_size = verneed_sec["sh_size"]

    # Walk the linked list; collect each entry's parsed contents and remember
    # whether it's our target.
    kept = []           # list of (vn_version, vn_file, list[(hash,flags,other,name)])
    target_others = set()
    found = False
    pos = section_offset
    while True:
        vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from(
            endian + "HHIII", data, pos
        )
        is_target = dynstr.get_string(vn_file) == _TARGET_LIB

        vnas = []
        aux_pos = pos + vn_aux
        for _ in range(vn_cnt):
            vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from(
                endian + "IHHII", data, aux_pos
            )
            if is_target:
                target_others.add(vna_other)
            else:
                vnas.append((vna_hash, vna_flags, vna_other, vna_name))
            if vna_next == 0:
                break
            aux_pos += vna_next

        if is_target:
            found = True
        else:
            kept.append((vn_version, vn_file, vnas))

        if vn_next == 0:
            break
        pos += vn_next

    if not found:
        return False

    # Rebuild the section bytes: each Verneed header followed by its Vernaux
    # entries, contiguously, terminated by vn_next=0 / vna_next=0.
    new_data = bytearray()
    for i, (vn_version, vn_file, vnas) in enumerate(kept):
        is_last_vn = i == len(kept) - 1
        vn_total = _VN_HDR_SIZE + _VNA_SIZE * len(vnas)
        vn_next = 0 if is_last_vn else vn_total
        new_data += struct.pack(
            endian + "HHIII",
            vn_version, len(vnas), vn_file, _VN_HDR_SIZE, vn_next,
        )
        for j, (h, fl, ot, nm) in enumerate(vnas):
            is_last_vna = j == len(vnas) - 1
            vna_next = 0 if is_last_vna else _VNA_SIZE
            new_data += struct.pack(
                endian + "IHHII", h, fl, ot, nm, vna_next,
            )

    if len(new_data) > section_size:
        raise RuntimeError(
            f"{so_path}: rebuilt .gnu.version_r ({len(new_data)} B) "
            f"exceeds original section size ({section_size} B)"
        )
    new_data += b"\x00" * (section_size - len(new_data))
    data[section_offset : section_offset + section_size] = new_data

    # Decrement DT_VERNEEDNUM in the dynamic table.
    is_64 = elf.elfclass == 64
    dyn_entry_size = 16 if is_64 else 8
    dyn_pack = endian + ("qq" if is_64 else "ii")
    dyn_sec = elf.get_section_by_name(".dynamic")
    if dyn_sec is not None:
        dyn_offset = dyn_sec["sh_offset"]
        dyn_size = dyn_sec["sh_size"]
        for i in range(0, dyn_size, dyn_entry_size):
            d_tag, d_val = struct.unpack_from(dyn_pack, data, dyn_offset + i)
            if d_tag == _DT_VERNEEDNUM:
                struct.pack_into(dyn_pack, data, dyn_offset + i, d_tag, d_val - 1)
                break
            if d_tag == 0:  # DT_NULL
                break

    # Rewrite DT_VERSYM entries that referenced our dropped Vernaux indices.
    versym_sec = elf.get_section_by_name(".gnu.version")
    if versym_sec is not None:
        vs_offset = versym_sec["sh_offset"]
        vs_size = versym_sec["sh_size"]
        for i in range(0, vs_size, 2):
            (raw,) = struct.unpack_from(endian + "H", data, vs_offset + i)
            if (raw & ~_VERSYM_HIDDEN_BIT) in target_others:
                new_raw = (raw & _VERSYM_HIDDEN_BIT) | _VER_NDX_GLOBAL
                struct.pack_into(endian + "H", data, vs_offset + i, new_raw)

    with open(so_path, "wb") as f:
        f.write(data)
    return True


def _iter_so_files(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for pattern in ("*.so", "*.so.*"):
                yield from sorted(p.rglob(pattern))
        elif p.is_file():
            yield p


def main(paths) -> None:
    seen: set[Path] = set()
    for so in _iter_so_files(paths):
        so = so.resolve()
        if so in seen:
            continue
        seen.add(so)
        if _strip_one(so):
            print(f"strip_zlib_ng_symbol_versions: {so}")


if __name__ == "__main__":
    main(sys.argv[1:])
