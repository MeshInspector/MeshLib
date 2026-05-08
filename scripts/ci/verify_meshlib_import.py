"""
Sanity-check that the freshly-built meshlib Python bindings load cleanly.

Run this with PYTHONPATH pointing at the build's bin directory (so that the
`meshlib/` package directory is discoverable). Exits non-zero with a real
Python traceback if any submodule import fails — catches problems like
missing PyInit_*, wrong libc++/libpython ABI, binding-generation
regressions, etc. before MRTest's embedded-python smoke test or pytest
collection bury them under CPython's opaque "ImportError: initialization
failed".

Iterates over every submodule shipped in the `meshlib/` package
directory (`mrmeshpy`, `mrmeshnumpy`, `mrviewerpy`, `mrcudapy`, …) so a
crash in any one of them is surfaced with its real traceback. This
mirrors what `test_python/helper/__init__.py` does at pytest
collection-time (it imports `meshlib.mrmeshpy` *and*
`meshlib.mrmeshnumpy`), so a successful run here means pytest's
collection-time imports won't trip either.
"""

import importlib
import os
import sys
import traceback


def discover_submodules(pkg) -> list[str]:
    """Find native-extension submodules in the meshlib package dir.

    Returns submodule names like 'mrmeshpy', 'mrmeshnumpy' — extracted
    from filenames matching `{name}.{so,dylib,pyd}`.
    """
    pkg_dir = os.path.dirname(pkg.__file__)
    extensions = ('.so', '.dylib', '.pyd')
    out = []
    for entry in sorted(os.listdir(pkg_dir)):
        for ext in extensions:
            if entry.endswith(ext):
                stem = entry[: -len(ext)]
                # Strip ABI tags like "mrmeshpy.cpython-310-darwin.so" → "mrmeshpy".
                if '.' in stem:
                    stem = stem.split('.', 1)[0]
                if stem and stem not in out and not stem.startswith('libpybind11nonlimitedapi'):
                    out.append(stem)
                break
    return out


def main() -> int:
    print('python:    ', sys.version)
    print('exe:       ', sys.executable)
    print('prefix:    ', sys.prefix)
    print('PYTHONPATH:', os.environ.get('PYTHONPATH', '<unset>'))
    print('---', flush=True)

    try:
        import meshlib  # noqa: F401
    except BaseException:
        print('FAIL: import meshlib', flush=True)
        traceback.print_exc()
        return 2
    print('OK: meshlib at', meshlib.__file__, flush=True)

    submodules = discover_submodules(meshlib)
    if not submodules:
        print('FAIL: no native submodules discovered alongside meshlib/__init__.py', flush=True)
        return 4
    print('Submodules to import:', submodules, flush=True)

    failed: list[str] = []
    for name in submodules:
        full = f'meshlib.{name}'
        try:
            mod = importlib.import_module(full)
        except BaseException:
            print(f'FAIL: import {full}', flush=True)
            traceback.print_exc()
            failed.append(name)
            continue
        loc = getattr(mod, '__file__', '<built-in>')
        print(f'OK:   {full} at {loc}', flush=True)

    if failed:
        print(f'\nSummary: {len(failed)}/{len(submodules)} submodule imports failed: {failed}', flush=True)
        return 3
    print(f'\nSummary: {len(submodules)}/{len(submodules)} submodules imported OK', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
