#!/usr/bin/env python3
"""Namespace the Emscripten port libraries MeshLib links against (freetype,
libpng, zlib), and the lz4 that c-blosc bundles, so the C# wasm package can
ship them without colliding with a host engine's own copies of the same
libraries.

Unity's WebGL player statically links its own trimmed freetype, libpng, zlib
and lz4 whose leaked globals collide with full copies of the same libraries,
and its frozen emsdk cache cannot build the ports on the consumer's machine.

Every symbol the listed libraries export is renamed with PREFIX, and the
package archives' references to them are rewritten to match, so nothing
about these libraries stays visible under a standard name. All libraries of
one package must be renamed by a single invocation: they share one rename
map, so references between them (libpng calls zlib) stay consistent. The
package's references to a --private-lib are left alone: that library serves
only the other renamed ones - used for zlib, whose standard names the package
already resolves elsewhere (libgdcmzlib.a ships them).

The renaming rewrites the wasm object files inside the archives the MeshLib
build already produced, so the shipped code is bit-identical to what was
built and tested, and nothing needs to know the ports' source layout or
compile flags. llvm-objcopy does not support symbol renaming for wasm
objects, but the format makes renaming exact: symbol names live only in the
`linking` custom section's symbol table and in the import entries' field
names, while relocations reference symbols by index, so rewriting the name
strings is complete and safe.
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


def splice(payload, spans):
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
    return splice(payload, spans)


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
    return splice(payload, spans)


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


def ar_members(path):
    """(name, data) of each object in an ar archive, duplicates included.

    The symbol index is dropped (llvm-ar rebuilds it); GNU long-name table
    entries and BSD inline long names are both resolved."""
    data = path.read_bytes()
    if data[:8] != b'!<arch>\n':
        sys.exit(f'{path.name}: not an ar archive')
    pos = 8
    longnames = b''
    members = []
    while pos + 60 <= len(data):
        hdr = data[pos:pos + 60]
        if hdr[58:60] != b'`\n':
            sys.exit(f'{path.name}: bad ar member header at offset {pos}')
        raw = hdr[:16].rstrip()
        size = int(hdr[48:58])
        body = data[pos + 60:pos + 60 + size]
        pos += 60 + size + (size & 1)
        if raw in (b'/', b'/SYM64/', b'__.SYMDEF', b'__.SYMDEF SORTED'):
            continue
        if raw == b'//':
            longnames = body
            continue
        if raw.startswith(b'#1/'):
            n = int(raw[3:])
            name, body = body[:n].rstrip(b'\0'), body[n:]
        elif raw.startswith(b'/'):
            off = int(raw[1:])
            name = longnames[off:longnames.index(b'/\n', off)]
        else:
            name = raw.rstrip(b'/')
        members.append((name.decode(), body))
    return members


def rewrite_archive(ar, src, dst, rename, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (name, data) in enumerate(ar_members(src)):
        # one directory per member keeps duplicate member names apart;
        # llvm-ar stores basenames, so the archive keeps the original names
        sub = workdir / f'{i:05d}'
        sub.mkdir(exist_ok=True)
        rewritten = rewrite_object(data, rename)
        rewrite_object(rewritten, set())  # self-check: the output must re-parse
        (sub / name).write_bytes(rewritten)
        paths.append(f'{sub.name}/{name}')
    rsp = workdir / 'members.rsp'
    rsp.write_text(''.join(p + '\n' for p in paths))
    tmp = workdir / 'out.a'
    if tmp.exists():
        tmp.unlink()
    run([ar, 'qc', str(tmp), '@' + rsp.name], cwd=workdir)
    tmp.replace(dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llvm-bin', default='', help='directory with llvm-nm and llvm-ar (default: from PATH)')
    p.add_argument('--package-dir', required=True,
                   help='directory with the MeshLib .a files being shipped; archives referencing the '
                        'renamed libraries are rewritten in place')
    p.add_argument('--vanilla-libs', required=True, help='dir where library stems are looked up (the emsdk cache lib dir for ports)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--pthread', action='store_true', help='prefer the -mt port variants (multithreaded package)')
    p.add_argument('--lib', action='append', default=[], metavar='STEM[=OUT]',
                   help='library to rename, as a stem globbed in --vanilla-libs (or a path to an archive); '
                        'written to OUT-mrml.a (default: STEM-mrml.a); the package references are rewritten; repeatable')
    p.add_argument('--private-lib', action='append', default=[], metavar='STEM[=OUT]',
                   help='like --lib, but the package references to it are left alone: it serves only the '
                        'other renamed libraries')
    args = p.parse_args()
    if not args.lib and not args.private_lib:
        p.error('at least one --lib/--private-lib is required')

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

    def resolve(spec):
        stem, _, out_name = spec.partition('=')
        path = Path(stem)
        src = path if path.suffix == '.a' and path.exists() else vanilla_lib(stem)
        return (out_name or stem) + '-mrml.a', src

    src_libs = {}
    d_public = set()  # exports whose package references get rewritten
    d_all = set()
    for spec in args.lib + args.private_lib:
        name, src = resolve(spec)
        src_libs[name] = src
        defined = nm_symbols(nm, src, '--defined-only')
        d_all |= defined
        if spec in args.lib:
            d_public |= defined

    renamed_srcs = {s.resolve() for s in src_libs.values()}
    package = [a for a in sorted(Path(args.package_dir).glob('*.a'))
               if not a.name.endswith('-mrml.a') and a.resolve() not in renamed_srcs]
    consumers = []
    refs = set()
    for a in package:
        used = (nm_symbols(nm, a, '--undefined-only') | nm_symbols(nm, a, '--defined-only')) & d_public
        if used:
            consumers.append(a)
            refs |= used

    (out / 'renamed_symbols.txt').write_text(''.join(s + '\n' for s in sorted(d_all)))
    print(f'renamed: {len(d_all)} symbols of {len(src_libs)} libraries')
    print(f'package references rewritten: {len(refs)} symbols in {len(consumers)} archives')
    for s in sorted(refs):
        print(f'  {s}')

    for name, src in src_libs.items():
        print(f'{src.name} -> {name}')
        rewrite_archive(ar, src, out / name, d_all, out / ('members-' + name[:-2]))
    for a in consumers:
        print(f'{a.name}: rewriting references in place')
        rewrite_archive(ar, a, a, refs, out / ('members-' + a.stem))

    # Verify with an independent tool: the renamed libraries export nothing
    # outside PREFIX and reference no renamed symbol by its old name, and the
    # package no longer uses any standard name of the public libraries.
    ok = True
    for lib in src_libs:
        bad = {s for s in nm_symbols(nm, out / lib, '--defined-only') if not s.startswith(PREFIX)}
        if bad:
            ok = False
            print(f'ERROR: {lib} exports unrenamed symbols: {sorted(bad)}')
        leaked = nm_symbols(nm, out / lib, '--undefined-only') & d_all
        if leaked:
            ok = False
            print(f'ERROR: {lib} references renamed symbols by their old names: {sorted(leaked)}')
    for a in consumers:
        leaked = (nm_symbols(nm, a, '--undefined-only') | nm_symbols(nm, a, '--defined-only')) & d_public
        if leaked:
            ok = False
            print(f'ERROR: {a.name} still uses standard names: {sorted(leaked)}')
    if not ok:
        sys.exit(1)
    print('verified: no standard name of the renamed libraries remains in the package')


if __name__ == '__main__':
    main()
