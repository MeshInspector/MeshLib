import os
import re
import sys

from pybind11_stubgen.__init__ import main

if __name__ == '__main__':
    if 'PYBIND11_STUBGEN_PATH' in os.environ:
        os.add_dll_directory(os.environ['PYBIND11_STUBGEN_PATH'])
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
