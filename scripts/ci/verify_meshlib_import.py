"""
Sanity-check that `import meshlib.mrmeshpy` works, with a real traceback
on failure instead of CPython's opaque "ImportError: initialization failed".
Run with PYTHONPATH pointing at the build's bin directory.
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
