"""
Sanity-check that the freshly-built meshlib Python bindings load cleanly.

Run with PYTHONPATH pointing at the build's bin directory. Exits non-zero
with a real Python traceback if `import meshlib` or
`import meshlib.mrmeshpy` fails — catches missing PyInit_*, wrong
libc++/libpython ABI, and binding-generation regressions before pytest's
collection (or MRTest's embedded-python smoke test) buries them under
CPython's opaque "ImportError: initialization failed".

Only `mrmeshpy` is checked here: it's the one whose load failure has
been the silent failure mode we hit. Other submodules pull in external
runtime deps (`mrmeshnumpy` requires `numpy`, which isn't installed in
some CI envs like manylinux's bare system Python) or trigger
pre-existing shutdown bugs (`mrviewerpy`'s CommandLoop destructor
asserts on a non-empty queue), and exercising them here produces false
positives. Their actual load is still covered downstream by pytest's
test collection (which runs in an env with proper deps).
"""

import os
import sys
import traceback


def main() -> int:
    print('python:    ', sys.version)
    print('exe:       ', sys.executable)
    print('prefix:    ', sys.prefix)
    print('PYTHONPATH:', os.environ.get('PYTHONPATH', '<unset>'))
    print('---')

    try:
        import meshlib  # noqa: F401
    except BaseException:
        print('FAIL: import meshlib')
        traceback.print_exc()
        return 2
    print('OK: meshlib at', meshlib.__file__)

    try:
        import meshlib.mrmeshpy as _mrmeshpy
    except BaseException:
        print('FAIL: import meshlib.mrmeshpy')
        traceback.print_exc()
        return 3
    print('OK: meshlib.mrmeshpy at', _mrmeshpy.__file__)
    return 0


if __name__ == '__main__':
    sys.exit(main())
