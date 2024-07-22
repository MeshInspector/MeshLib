from pathlib import Path
import meshlib.mrmeshpy as mlib
from constants import test_files_path
import pytest
from helpers.meshlib_helpers import compare_points_similarity, compare_meshes_similarity


@pytest.mark.parametrize("filename", [
    "points_0.0.128.21.mru",
    "points_0.1.0.0.mru",
    "points_2.1.1.44.mru",
    "points_2.3.0.17.mru",
    "points_2.4.3.45.mru",
])
def test_open_old_points(filename):
    """
    Loads .mru files with pointCloud, extract pointCloud from scene and compare extracted object with the reference one
    """
    ref_file = "ref.ctm"
    input_folder = Path(test_files_path) / "old_files" / "mru" / "points"
    scene_obj = mlib.loadSceneObject(input_folder / filename)
    point_cloud = scene_obj.children()[0].extractPoints()

    refPointCloud = mlib.loadPoints(input_folder / ref_file)
    compare_points_similarity(point_cloud, refPointCloud, verts_thresh=0, rhsdr_thresh=0.0001)


@pytest.mark.parametrize("filename", [
    "mesh_0.0.128.21.mru",
    "mesh_0.1.0.0.mru",
    "mesh_2.1.1.44.mru",
    "mesh_2.3.0.17.mru",
    "mesh_2.4.3.45.mru",
])
def test_open_old_meshes(filename):
    """
    Loads .mru files with pointCloud, extract pointCloud from scene and compare extracted object with the reference one
    """
    ref_file = "ref.ctm"
    input_folder = Path(test_files_path) / "old_files" / "mru" / "meshes"
    scene_obj = mlib.loadSceneObject(input_folder / filename)
    mesh = scene_obj.children()[0].extractMesh()
    ref = mlib.loadMesh(input_folder / ref_file)
    compare_meshes_similarity(mesh, ref, verts_thresh=0, rhsdr_thresh=0.0001)
