import os
import sys
import pytest
# insert 0 to find mrpy.so in working directory and not in system
sys.path.insert(0, os.environ["MeshLibPyModulesPath"])
import mrmeshpy as mrmesh
