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

import meshlib.mrmeshpy as mrmesh
import meshlib.mrmeshnumpy as mrmeshnumpy

# Check if we're using the new parsed bindings.
is_new_binding = True
try:
    mrmesh.UniformSamplingSettings
except AttributeError:
    is_new_binding = False
