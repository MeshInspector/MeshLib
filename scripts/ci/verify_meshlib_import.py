"""
Sanity-check that the freshly-built meshlib Python bindings load cleanly.

Run this with PYTHONPATH pointing at the build's bin directory (so that the
`meshlib/` package directory is discoverable). Exits non-zero with a real
Python traceback if any of the targeted submodule imports fails — catches
problems like missing PyInit_*, wrong libc++/libpython ABI,
binding-generation regressions, etc. before pytest's collection (or
MRTest's embedded-python smoke test) buries them under CPython's opaque
"ImportError: initialization failed".

The targeted submodule list mirrors what `test_python/helper/__init__.py`
imports at module load — that helper module is pulled in by every test
file, so its imports run during pytest collection, and a crash in any of
those imports kills the collector before pytest can report it. Other
submodules in the `meshlib/` package (e.g. `mrviewerpy`) are intentionally
NOT exercised here: importing `mrviewerpy` and letting Python shut down
trips a separate pre-existing assertion (CommandLoop destructor expecting
a drained command queue), which is unrelated to the
binding-load problem this script is meant to gate. If the helper file ever
imports more submodules, mirror them here.
"""

import importlib
import os
import sys
import traceback

# Keep in sync with test_python/helper/__init__.py.
HELPER_SUBMODULES = ('mrmeshpy', 'mrmeshnumpy')


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

    print('Submodules to import:', list(HELPER_SUBMODULES), flush=True)

    failed: list[str] = []
    for name in HELPER_SUBMODULES:
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
        print(f'\nSummary: {len(failed)}/{len(HELPER_SUBMODULES)} submodule imports failed: {failed}', flush=True)
        return 3
    print(f'\nSummary: {len(HELPER_SUBMODULES)}/{len(HELPER_SUBMODULES)} submodules imported OK', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
