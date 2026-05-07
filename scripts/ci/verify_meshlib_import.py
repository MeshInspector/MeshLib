"""
Sanity-check that the freshly-built meshlib Python bindings load cleanly.

Run this with PYTHONPATH pointing at the build's bin directory (so that the
`meshlib/` package directory is discoverable). Exits non-zero with a real
Python traceback if `import meshlib` or `import meshlib.mrmeshpy` fails —
catches problems like missing PyInit_*, wrong libc++/libpython ABI,
binding-generation regressions, etc. before MRTest's embedded-python smoke
test buries them under CPython's opaque "ImportError: initialization failed".
"""

import os
import sys
import traceback


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

    try:
        import meshlib.mrmeshpy as _mrmeshpy
    except BaseException:
        print('FAIL: import meshlib.mrmeshpy', flush=True)
        traceback.print_exc()
        return 3
    print('OK: meshlib.mrmeshpy at', _mrmeshpy.__file__, flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
