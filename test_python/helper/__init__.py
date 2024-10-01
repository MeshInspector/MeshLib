import os
import sys

import pytest

working_directory = str()
# insert 0 to find mrpy.so in working directory and not in system
if (
    "MeshLibPyModulesPath" in os.environ
    and not os.environ["MeshLibPyModulesPath"] in sys.path
):
    sys.path.insert(0, os.environ["MeshLibPyModulesPath"])
    working_directory = os.environ["MeshLibPyModulesPath"]

is_new_binding = True

if os.environ.get('USE_MESHLIB2_PY', '0') != '0':
    from meshlib2 import mrmeshpy as mrmesh
    from meshlib2 import mrmeshnumpy
else:
    if os.environ.get('OLD_MESHLIB_BINDINGS', '0') != '0':
        is_new_binding = False
    import meshlib.mrmeshpy as mrmesh
    import meshlib.mrmeshnumpy as mrmeshnumpy
