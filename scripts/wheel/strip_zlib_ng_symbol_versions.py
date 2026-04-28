#!/usr/bin/env python3
"""Neutralize the libz-ng.so.2 entry in .gnu.version_r of consumer .so files.

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

  1. Zero its vn_cnt. pyelftools (used by auditwheel) gates Vernaux
     iteration on vn_cnt; with vn_cnt=0 auditwheel sees no version
     requirements for libz-ng.

  2. Rewrite every .gnu.version (DT_VERSYM) entry that referenced one
     of this Verneed's Vernaux indices to VER_NDX_GLOBAL (1, "global,
     unversioned"). This keeps the dynamic linker from looking up
     version names against the now-empty Vernaux set when resolving
     zng_* symbols at load time -- it resolves them by name only.

The rest of the ELF -- DT_VERNEED, DT_VERNEEDNUM, .dynsym, the
underlying Vernaux byte chain itself -- is left intact, so other
versioned dependencies (GLIBC_*, GLIBCXX_*, libgcc_s, ...) are not
touched.

Run on every consumer .so before auditwheel.
"""

import io
import struct
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile
from elftools.elf.gnuversions import GNUVerNeedSection


_TARGET_LIB = "libz-ng.so.2"
_VER_NDX_GLOBAL = 1
_VERSYM_HIDDEN_BIT = 0x8000


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

    # Verneed layout (16 bytes, identical for ELF32/64):
    #   vn_version(2) vn_cnt(2) vn_file(4) vn_aux(4) vn_next(4)
    # Vernaux layout (16 bytes):
    #   vna_hash(4) vna_flags(2) vna_other(2) vna_name(4) vna_next(4)
    target_pos = -1
    target_others: set[int] = set()

    pos = verneed_sec["sh_offset"]
    while True:
        _, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from(
            endian + "HHIII", data, pos
        )
        if dynstr.get_string(vn_file) == _TARGET_LIB and vn_cnt != 0:
            target_pos = pos
            aux_pos = pos + vn_aux
            for _ in range(vn_cnt):
                _, _, vna_other, _, vna_next = struct.unpack_from(
                    endian + "IHHII", data, aux_pos
                )
                target_others.add(vna_other)
                if vna_next == 0:
                    break
                aux_pos += vna_next
            break
        if vn_next == 0:
            return False
        pos += vn_next

    # 1. vn_cnt = 0 (offset +2 in the Verneed header)
    struct.pack_into(endian + "H", data, target_pos + 2, 0)

    # 2. Rewrite DT_VERSYM entries pointing into our Vernaux set to GLOBAL.
    # The high bit of a versym entry is "hidden"; preserve it for sanity
    # though it should never be set on UND symbols.
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
