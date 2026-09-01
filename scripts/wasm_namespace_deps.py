#!/usr/bin/env python3
"""Ship namespaced (symbol-prefixed) copies of the Emscripten port libraries
MeshLib links against (freetype, libpng, zlib) in the C# wasm package, so
they cannot collide with a host engine's own copies of the same libraries.

Unity's WebGL player statically links its own trimmed freetype, libpng and
zlib whose leaked globals collide with full copies of the same libraries, and
its frozen emsdk cache cannot build the ports on the consumer's machine.

Every symbol the port libraries export is renamed with PREFIX, except the
exact set the MeshLib package archives reference (computed with llvm-nm), so
the already-compiled MeshLib objects link against these libraries unchanged.
zlib is treated as private to libpng: MeshLib's own zlib references keep
resolving to the archives that provide them today, never to this copy.

The renaming rewrites the wasm object files of the archives the MeshLib build
already produced, so the shipped code is bit-identical to what was built and
tested, and nothing needs to know the ports' source layout or compile flags.
llvm-objcopy does not support symbol renaming for wasm objects, but the
format makes renaming exact: symbol names live only in the `linking` custom
section's symbol table and in the import entries' field names, while
relocations reference symbols by index, so rewriting the name strings is
complete and safe.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PREFIX = 'mrml_'

IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# Wasm object file conventions: see tool-conventions/Linking.md.
WASM_SYM_UNDEFINED = 0x10
WASM_SYM_EXPLICIT_NAME = 0x40
KIND_FUNCTION, KIND_DATA, KIND_GLOBAL, KIND_SECTION, KIND_TAG, KIND_TABLE = range(6)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f'FAILED ({r.returncode}): {" ".join(map(str, cmd))}\n{r.stdout}\n{r.stderr}')
    return r.stdout


def nm_symbols(nm, archive, mode):
    out = run([nm, mode, '--extern-only', str(archive)])
    syms = set()
    for line in out.splitlines():
        parts = line.split()
        if parts and IDENT.match(parts[-1]):
            syms.add(parts[-1])
    return syms


def enc_leb(value):
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


class Reader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def u8(self):
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def leb(self):
        result = 0
        shift = 0
        while True:
            b = self.u8()
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7

    def str_span(self):
        # returns (text, start, end) where [start, end) covers the
        # length-prefixed string, so it can be spliced out
        start = self.pos
        n = self.leb()
        s = self.buf[self.pos:self.pos + n].decode()
        self.pos += n
        return s, start, self.pos

    def limits(self):
        flags = self.leb()
        self.leb()
        if flags & 1:
            self.leb()


def splice(payload, spans, rename):
    out = bytearray()
    last = 0
    for start, end, old in spans:
        new = (PREFIX + old).encode()
        out += payload[last:start]
        out += enc_leb(len(new)) + new
        last = end
    out += payload[last:]
    return bytes(out)


def rewrite_imports(payload, rename):
    # undefined symbols carry their name as the import entry's field name
    r = Reader(payload)
    spans = []
    for _ in range(r.leb()):
        r.str_span()  # module
        field, start, end = r.str_span()
        kind = r.u8()
        if kind == 0x00:
            r.leb()  # function: type index
        elif kind == 0x01:
            r.u8()
            r.limits()  # table: reftype, limits
        elif kind == 0x02:
            r.limits()  # memory
        elif kind == 0x03:
            r.u8()
            r.u8()  # global: valtype, mutability
        elif kind == 0x04:
            r.u8()
            r.leb()  # tag: attribute, type index
        else:
            sys.exit(f'unknown import kind {kind}')
        if field in rename:
            spans.append((start, end, field))
    if r.pos != len(payload):
        sys.exit('trailing bytes in the import section')
    return splice(payload, spans, rename)


def rewrite_symtab(payload, rename):
    r = Reader(payload)
    spans = []
    for _ in range(r.leb()):
        kind = r.u8()
        flags = r.leb()
        if kind in (KIND_FUNCTION, KIND_GLOBAL, KIND_TAG, KIND_TABLE):
            r.leb()  # index into the kind's index space
            # an undefined symbol's name lives in its import entry instead,
            # unless an explicit one is stored here
            if not (flags & WASM_SYM_UNDEFINED) or (flags & WASM_SYM_EXPLICIT_NAME):
                name, start, end = r.str_span()
                if name in rename:
                    spans.append((start, end, name))
        elif kind == KIND_DATA:
            name, start, end = r.str_span()
            if name in rename:
                spans.append((start, end, name))
            if not (flags & WASM_SYM_UNDEFINED):
                r.leb()
                r.leb()
                r.leb()  # segment, offset, size
        elif kind == KIND_SECTION:
            r.leb()
        else:
            sys.exit(f'unknown symbol kind {kind}')
    if r.pos != len(payload):
        sys.exit('trailing bytes in the symbol table')
    return splice(payload, spans, rename)


def rewrite_linking(content, rename):
    r = Reader(content)
    version = r.leb()
    if version != 2:
        sys.exit(f'unsupported linking section version {version}')
    out = bytearray(content[:r.pos])
    while r.pos < len(content):
        sub_type = r.u8()
        size = r.leb()
        payload = content[r.pos:r.pos + size]
        r.pos += size
        if sub_type == 8:  # WASM_SYMBOL_TABLE
            payload = rewrite_symtab(payload, rename)
        out.append(sub_type)
        out += enc_leb(len(payload))
        out += payload
    return bytes(out)


def rewrite_object(data, rename):
    if data[:8] != b'\0asm\x01\0\0\0':
        sys.exit('not a wasm object file (LTO bitcode members are not supported)')
    out = bytearray(data[:8])
    pos = 8
    while pos < len(data):
        sec_id = data[pos]
        r = Reader(data)
        r.pos = pos + 1
        size = r.leb()
        payload = data[r.pos:r.pos + size]
        pos = r.pos + size
        if sec_id == 2:
            payload = rewrite_imports(payload, rename)
        elif sec_id == 0:
            pr = Reader(payload)
            name, _, hdr_end = pr.str_span()
            if name == 'linking':
                payload = payload[:hdr_end] + rewrite_linking(payload[hdr_end:], rename)
            # relocation sections reference symbols by index and the `name`
            # section is debug info, so neither needs a fixup; section count
            # and order are preserved, keeping cross-section indices valid
        out.append(sec_id)
        out += enc_leb(len(payload))
        out += payload
    return bytes(out)


def rewrite_archive(ar, src, dst, rename, workdir):
    members = [m for m in run([ar, 't', str(src)]).splitlines() if m]
    if len(set(members)) != len(members):
        sys.exit(f'{src.name}: duplicate member names, flat extraction would clobber')
    workdir.mkdir(parents=True, exist_ok=True)
    run([ar, 'x', str(src)], cwd=workdir)
    for m in members:
        p = workdir / m
        rewritten = rewrite_object(p.read_bytes(), rename)
        rewrite_object(rewritten, set())  # self-check: the output must re-parse
        p.write_bytes(rewritten)
    if dst.exists():
        dst.unlink()
    run([ar, 'qc', str(dst)] + members, cwd=workdir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llvm-bin', default='', help='directory with llvm-nm and llvm-ar (default: from PATH)')
    p.add_argument('--package-dir', required=True, help='directory with the MeshLib .a files being shipped')
    p.add_argument('--vanilla-libs', required=True, help='dir with the port builds to rename (libfreetype*.a, libpng*.a, libz*.a)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--pthread', action='store_true', help='prefer the -mt port variants (multithreaded package)')
    args = p.parse_args()

    bindir = Path(args.llvm_bin) if args.llvm_bin else None

    def tool(name):
        if not bindir:
            return name
        for cand in (bindir / name, bindir / (name + '.exe')):
            if cand.exists():
                return str(cand)
        sys.exit(f'{name} not found in {bindir}')

    nm = tool('llvm-nm')
    ar = tool('llvm-ar')
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vanilla = Path(args.vanilla_libs)

    def vanilla_lib(stem):
        # prefer the variant matching the threading mode; port builds may
        # carry suffixes (-mt, -wasm-sjlj, ...) or none at all
        cands = sorted(vanilla.glob(stem + '*.a'),
                       key=lambda c: (('-mt' in c.name) != args.pthread, len(c.name)))
        if not cands:
            sys.exit(f'no {stem}*.a in {vanilla}; the MeshLib build should have built the ports')
        return cands[0]

    src_libs = {
        'libfreetype-mrml.a': vanilla_lib('libfreetype'),
        'libpng-mrml.a': vanilla_lib('libpng'),
        'libzlib-mrml.a': vanilla_lib('libz'),
    }

    d_ft = nm_symbols(nm, src_libs['libfreetype-mrml.a'], '--defined-only')
    d_png = nm_symbols(nm, src_libs['libpng-mrml.a'], '--defined-only')
    d_z = nm_symbols(nm, src_libs['libzlib-mrml.a'], '--defined-only')

    u_pkg = set()
    for a in sorted(Path(args.package_dir).glob('*.a')):
        if a.name.endswith('-mrml.a'):
            continue  # a previous run's own outputs
        u_pkg |= nm_symbols(nm, a, '--undefined-only')

    # The unrenamed surface: exactly what the package references from freetype
    # and libpng. zlib stays fully renamed - it exists only to serve libpng.
    s_api = u_pkg & (d_ft | d_png)
    rename = (d_ft | d_png | d_z) - s_api

    (out / 'renamed_symbols.txt').write_text(''.join(s + '\n' for s in sorted(rename)))
    print(f'kept unrenamed (referenced by the package): {len(s_api)}')
    for s in sorted(s_api):
        print(f'  {s}')
    print(f'renamed: {len(rename)}')

    for name, src in src_libs.items():
        print(f'{src.name} -> {name}')
        rewrite_archive(ar, src, out / name, rename, out / ('members-' + name[:-2]))

    # Verify with an independent tool: no library may export a name outside
    # PREFIX except the s_api set, and nothing may still reference a renamed
    # symbol by its old name.
    ok = True
    for lib in src_libs:
        bad = {s for s in nm_symbols(nm, out / lib, '--defined-only') if not s.startswith(PREFIX)} - s_api
        if bad:
            ok = False
            print(f'ERROR: {lib} exports unrenamed symbols: {sorted(bad)}')
        leaked = nm_symbols(nm, out / lib, '--undefined-only') & rename
        if leaked:
            ok = False
            print(f'ERROR: {lib} references renamed symbols by their old names: {sorted(leaked)}')
    if not ok:
        sys.exit(1)
    print('verified: only the package-referenced API is exported unrenamed')


if __name__ == '__main__':
    main()
