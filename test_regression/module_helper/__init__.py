import os
import sys

# insert 0 to find mrpy.so in working directory and not in system
if "MeshLibPyModulesPath" in os.environ and not os.environ["MeshLibPyModulesPath"] in sys.path:
    sys.path.insert(0, os.environ["MeshLibPyModulesPath"])

# Check if we're using the new parsed bindings.
is_new_binding = True
try:
    mrmesh.UniformSamplingSettings
except NameError:
    is_new_binding = False

import meshlib.mrmeshpy as mrmeshpy
import meshlib.mrmeshnumpy as mrmeshnumpy
