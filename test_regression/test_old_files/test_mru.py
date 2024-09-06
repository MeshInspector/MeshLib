from pathlib import Path
from constants import test_files_path
import pytest
from module_helper import *
from helpers.meshlib_helpers import (compare_points_similarity, compare_meshes_similarity, compare_voxels,
                                     compare_lines, compare_distance_maps)


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
    scene_obj = mrmeshpy.loadSceneObject(input_folder / filename)
    point_cloud = scene_obj.children()[0].extractPoints()

    refPointCloud = mrmeshpy.loadPoints(input_folder / ref_file)
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
    scene_obj = mrmeshpy.loadSceneObject(input_folder / filename)
    mesh = scene_obj.children()[0].extractMesh()
    ref = mrmeshpy.loadMesh(input_folder / ref_file)
    compare_meshes_similarity(mesh, ref, verts_thresh=0, rhsdr_thresh=0.0001)


@pytest.mark.parametrize("filename", [
    "voxels_0.0.128.21.mru",
    "voxels_0.1.0.0.mru",
    "voxels_2.1.1.44.mru",
    "voxels_2.2.0.79.mru",
    "voxels_2.3.0.17.mru",
    "voxels_2.4.3.65.mru",
])
def test_open_old_voxels(filename):
    ref_file = "W74_H91_S124_V921_921_921_G0_F reff.raw"
    input_folder = Path(test_files_path) / "old_files" / "mru" / "voxels"
    scene_obj = mrmeshpy.loadSceneObject(input_folder / filename)
    voxels = scene_obj.children()[0].extractVoxels()

    compare_voxels(voxels, input_folder / ref_file)


@pytest.mark.parametrize("filename", [
    "lines_0.0.128.21.mru",
    "lines_0.1.0.0.mru",
    "lines_2.1.1.44.mru",
    "lines_2.2.0.79.mru",
    "lines_2.3.0.17.mru",
    "lines_2.4.3.65.mru",
])
def test_open_old_lines(filename):
    ref_file = "ref_IsoLines2.mrlines"
    input_folder = Path(test_files_path) / "old_files" / "mru" / "lines"
    scene_obj = mrmeshpy.loadSceneObject(input_folder / filename)
    lines = scene_obj.children()[0].extractLines()

    ref_lines = mrmeshpy.loadLines(input_folder / ref_file)
    # assert ref_lines.computeBoundingBox().diagonal() == lines.computeBoundingBox().diagonal()
    compare_lines(input_folder / ref_file, lines)


@pytest.mark.parametrize("filename", [
    "dm_0.0.128.21.mru",
    "dm_0.1.0.0.mru",
    "dm_2.1.1.144.mru",
    "dm_2.2.0.79.mru",
    "dm_2.3.0.17.mru",
])
def test_open_old_dm(filename):
    ref_file = "ref.mru"
    input_folder = Path(test_files_path) / "old_files" / "mru" / "distance_maps"

    scene_obj_ref = mrmeshpy.loadSceneObject(input_folder / ref_file)
    dm_ref = scene_obj_ref.children()[0].extractDistanceMap()

    scene_obj = mrmeshpy.loadSceneObject(input_folder / filename)
    dm = scene_obj.children()[0].extractDistanceMap()

    compare_distance_maps(dm, dm_ref)
