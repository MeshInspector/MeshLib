import pytest

from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='positive'",
    reason="Only run when --run-cuda is 'positive'",
)
def test_cuda_mesh_to_dm(cuda_module, tmp_path):

    #  Load input point
    case_name = f"cuda-mesh-to-dm"
    input_folder = Path(test_files_path) / "cuda" / "distance_map"
    mesh = mrmeshpy.loadMesh(input_folder / "input.ctm")

    # Create distance map
    params = mrmeshpy.MeshToDistanceMapParams()
    params.direction = mrmeshpy.Vector3f(x=0, y=1, z=0)
    params.resolution = mrmeshpy.Vector2i(x=1000, y=1000)
    params.orgPoint = mrmeshpy.Vector3f(x=0, y=-125, z=-20)
    params.xRange = mrmeshpy.Vector3f(x=150, y=150, z=0)
    params.yRange = mrmeshpy.Vector3f(x=0, y=150, z=150)

    dm = cuda_module.computeDistanceMap(mesh=mesh, params=params)
    mrmeshpy.saveDistanceMapToImage(distMap=dm, filename=tmp_path / "a.png")

    # === Verification
    map = mrmeshpy.loadDistanceMapFromImage(tmp_path / "a.png")
    aff = mrmeshpy.AffineXf3f()
    mesh = mrmeshpy.distanceMapToMesh(map, aff)
    mrmeshpy.saveMesh(mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh, skip_volume=True)

    with check:
        # rough check of calculated memory usage
        assert 30000000 < cuda_module.computeDistanceMapHeapBytes(mesh=mesh, params=params) < 50000000

