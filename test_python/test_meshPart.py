import pytest
from helper import *

# This tests the bugfix for: https://github.com/MeshInspector/MeshInspectorCode/issues/6440
# Also see: https://github.com/MeshInspector/mrbind/commit/1bd3695342c6cb448120af2a7e67d6eda7239ef3
def test_mesh_part():
    mr_mesh = mrmesh.makeCube()

    p = mrmesh.MeshPart(mr_mesh)
    mr_mesh = None # <-- this causes the bug
    params = mrmesh.OffsetParameters()
    params.voxelSize = mrmesh.suggestVoxelSize(p, 5e3)
    print(params.voxelSize)

    mr_mesh = mrmesh.offsetMesh(mp=p, offset=float(2), params=params)
