#!/usr/bin/env python3
"""Build namespaced (symbol-prefixed) static libraries of the Emscripten ports
MeshLib links against (freetype, libpng, zlib) so they can be shipped in the
C# wasm package without colliding with a host engine's own copies.

Unity's WebGL player statically links its own trimmed freetype, libpng and
zlib whose leaked globals collide with full copies of the same libraries, and
its frozen emsdk cache cannot build ports on the consumer's machine. Renaming
is done at compile time (a generated header of #define lines) because
llvm-objcopy does not support symbol renaming for wasm objects.

Every symbol a dependency exports is renamed with PREFIX, except the exact
set the MeshLib package archives reference (computed with llvm-nm), so the
already-compiled MeshLib objects link against these libraries unchanged.
zlib is treated as private to libpng: MeshLib's own zlib references keep
resolving to the archives that provide them today, never to this copy.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PREFIX = 'mrml_'

FREETYPE_SRCS = [
    'src/autofit/autofit.c', 'src/base/ftadvanc.c', 'src/base/ftbbox.c',
    'src/base/ftbdf.c', 'src/base/ftbitmap.c', 'src/base/ftcalc.c',
    'src/base/ftcid.c', 'src/base/ftdbgmem.c', 'src/base/ftdebug.c',
    'src/base/ftfntfmt.c', 'src/base/ftfstype.c', 'src/base/ftgasp.c',
    'src/base/ftgloadr.c', 'src/base/ftglyph.c', 'src/base/ftgxval.c',
    'src/base/ftinit.c', 'src/base/ftlcdfil.c', 'src/base/ftmm.c',
    'src/base/ftobjs.c', 'src/base/ftotval.c', 'src/base/ftoutln.c',
    'src/base/ftpatent.c', 'src/base/ftpfr.c', 'src/base/ftrfork.c',
    'src/base/ftsnames.c', 'src/base/ftstream.c', 'src/base/ftstroke.c',
    'src/base/ftsynth.c', 'src/base/ftsystem.c', 'src/base/fttrigon.c',
    'src/base/fttype1.c', 'src/base/ftutil.c', 'src/base/ftwinfnt.c',
    'src/bdf/bdf.c', 'src/bzip2/ftbzip2.c', 'src/cache/ftcache.c',
    'src/cff/cff.c', 'src/cid/type1cid.c', 'src/gzip/ftgzip.c',
    'src/lzw/ftlzw.c', 'src/pcf/pcf.c', 'src/pfr/pfr.c',
    'src/psaux/psaux.c', 'src/pshinter/pshinter.c', 'src/psnames/psmodule.c',
    'src/raster/raster.c', 'src/sfnt/sfnt.c', 'src/smooth/smooth.c',
    'src/truetype/truetype.c', 'src/type1/type1.c', 'src/type42/type42.c',
    'src/winfonts/winfnt.c',
]

ZLIB_SRCS = [
    'adler32.c', 'compress.c', 'crc32.c', 'deflate.c', 'gzclose.c',
    'gzlib.c', 'gzread.c', 'gzwrite.c', 'infback.c', 'inffast.c',
    'inflate.c', 'inftrees.c', 'trees.c', 'uncompr.c', 'zutil.c',
]

IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# Definition sites the rename header cannot reach: libpng defines these with
# the parenthesized "(PNGAPI name)" idiom that defeats macro expansion, and
# zlib #undef's gzgetc right before defining it. Patched in source copies.
PATCHES = {
    'pngrutil.c': [
        ('png_get_uint_32)(png_const_bytep buf)', f'{PREFIX}png_get_uint_32)(png_const_bytep buf)'),
        ('png_get_int_32)(png_const_bytep buf)', f'{PREFIX}png_get_int_32)(png_const_bytep buf)'),
        ('png_get_uint_16)(png_const_bytep buf)', f'{PREFIX}png_get_uint_16)(png_const_bytep buf)'),
    ],
    'gzread.c': [
        ('int ZEXPORT gzgetc(file)', f'int ZEXPORT {PREFIX}gzgetc(file)'),
        ('    return gzgetc(file);', f'    return {PREFIX}gzgetc(file);'),
    ],
}


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


def compile_lib(emcc, ar, srcs, src_root, out_lib, extra_flags, jobs_dir):
    jobs_dir.mkdir(parents=True, exist_ok=True)
    objs = []
    for src in srcs:
        path = src_root / src
        name = Path(src).name
        if name in PATCHES:
            text = path.read_text()
            for old, new in PATCHES[name]:
                assert text.count(old) == 1, f'{name}: pattern not found once: {old}'
                text = text.replace(old, new)
            path = jobs_dir / name
            path.write_text(text)
        obj = jobs_dir / (src.replace('/', '_') + '.o')
        run([emcc, '-c', str(path), '-o', str(obj), '-O2', '-I' + str(src_root)] + extra_flags)
        objs.append(str(obj))
    if out_lib.exists():
        out_lib.unlink()
    run([ar, 'qc', str(out_lib)] + objs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--emscripten', required=True, help='emscripten root (contains emcc, cache/)')
    p.add_argument('--package-dir', required=True, help='directory with the MeshLib .a files being shipped')
    p.add_argument('--ports-dir', required=True, help='emsdk cache ports dir with unpacked freetype/libpng/zlib sources')
    p.add_argument('--vanilla-libs', required=True, help='dir with unrenamed port builds (libfreetype.a, libpng.a, libz.a) used to enumerate their exports')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--pthread', action='store_true', help='build with -pthread (multithreaded package variant)')
    args = p.parse_args()

    em = Path(args.emscripten)
    llvm = em.parent / 'llvm'
    nm = str(llvm / 'llvm-nm.exe') if (llvm / 'llvm-nm.exe').exists() else 'llvm-nm'
    ar = str(llvm / 'llvm-ar.exe') if (llvm / 'llvm-ar.exe').exists() else 'llvm-ar'
    emcc = str(em / ('emcc.bat' if os.name == 'nt' else 'emcc'))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vanilla = Path(args.vanilla_libs)
    ports = Path(args.ports_dir)

    def vanilla_lib(stem):
        # port builds may carry variant suffixes (-mt, -wasm-sjlj, ...);
        # any variant works here since only symbol names are read
        cands = sorted(vanilla.glob(stem + '*.a'), key=lambda c: len(c.name))
        if not cands:
            sys.exit(f'no {stem}*.a in {vanilla}; build the ports first (embuilder build freetype libpng zlib)')
        return cands[0]

    d_ft = nm_symbols(nm, vanilla_lib('libfreetype'), '--defined-only')
    d_png = nm_symbols(nm, vanilla_lib('libpng'), '--defined-only')
    d_z = nm_symbols(nm, vanilla_lib('libz'), '--defined-only')

    u_pkg = set()
    for a in sorted(Path(args.package_dir).glob('*.a')):
        u_pkg |= nm_symbols(nm, a, '--undefined-only')

    # The unrenamed surface: exactly what the package references from freetype
    # and libpng. zlib stays fully renamed - it exists only to serve libpng.
    s_api = u_pkg & (d_ft | d_png)
    rename = (d_ft | d_png | d_z) - s_api

    hdr = out / 'mrml_rename.h'
    with open(hdr, 'w', newline='\n') as f:
        f.write('/* generated by wasm_namespace_deps.py - do not edit */\n')
        f.write('#pragma once\n')
        for s in sorted(rename):
            f.write(f'#define {s} {PREFIX}{s}\n')

    print(f'kept unrenamed (referenced by the package): {len(s_api)}')
    for s in sorted(s_api):
        print(f'  {s}')
    print(f'renamed: {len(rename)}')

    ft_src = next(ports.glob('freetype/FreeType-*'))
    png_src = next(ports.glob('libpng/libpng-*'))
    z_src = next(ports.glob('zlib/zlib-*'))
    missing = [f for f in FREETYPE_SRCS if not (ft_src / f).exists()]
    if missing:
        sys.exit(f'freetype port layout changed ({ft_src.name}); update FREETYPE_SRCS: {missing}')

    inc = ['-include', str(hdr)]
    if args.pthread:
        inc.append('-pthread')
    compile_lib(emcc, ar, ZLIB_SRCS, z_src, out / 'libzlib-mrml.a',
                inc + ['-Wno-deprecated-non-prototype'], out / 'obj-z')
    png_srcs = sorted(f.name for f in png_src.glob('*.c') if f.name != 'pngtest.c')
    compile_lib(emcc, ar, png_srcs, png_src, out / 'libpng-mrml.a',
                inc + ['-I' + str(z_src)], out / 'obj-png')
    compile_lib(emcc, ar, FREETYPE_SRCS, ft_src, out / 'libfreetype-mrml.a',
                inc + ['-DFT2_BUILD_LIBRARY', '-I' + str(ft_src / 'include')], out / 'obj-ft')

    # Verify: no library may export a name outside PREFIX except the s_api set.
    ok = True
    for lib in ('libfreetype-mrml.a', 'libpng-mrml.a', 'libzlib-mrml.a'):
        exported = nm_symbols(nm, out / lib, '--defined-only')
        bad = {s for s in exported if not s.startswith(PREFIX)} - s_api
        if bad:
            ok = False
            print(f'ERROR: {lib} exports unrenamed symbols: {sorted(bad)}')
    # And libpng's zlib references must all be renamed (private zlib).
    png_undef = nm_symbols(nm, out / 'libpng-mrml.a', '--undefined-only')
    leaked = png_undef & d_z
    if leaked:
        ok = False
        print(f'ERROR: libpng-mrml still references unrenamed zlib: {sorted(leaked)}')
    if not ok:
        sys.exit(1)
    print('verified: only the package-referenced API is exported unrenamed')


if __name__ == '__main__':
    main()
