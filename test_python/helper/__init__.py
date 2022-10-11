import os
import sys
import pytest

# insert 0 to find mrpy.so in working directory and not in system
if "MeshLibPyModulesPath" in os.environ and not os.environ["MeshLibPyModulesPath"] in sys.path:
    sys.path.insert(0, os.environ["MeshLibPyModulesPath"])

import meshlib.mrmeshpy as mrmesh
