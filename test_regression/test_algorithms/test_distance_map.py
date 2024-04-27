from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mlib

import pytest


@pytest.mark.smoke
def test_mesh_to_distance_map_to_mesh(tmp_path):
    """
    Load mesh, convert to distance map, save to image, load from image, convert to mesh, compare with reference
    """
    #  Load input point
    case_name = f"mesh-to-dm-to-mesh"
    input_folder = Path(test_files_path) / "algorithms" / "distance_map"
    mesh = mlib.loadMesh(input_folder / "input.ctm")

    # Create distance map
    params = mlib.MeshToDistanceMapParams()
    params.direction = mlib.Vector3f(x=0, y=1, z=0)
    params.resolution = mlib.Vector2i(x=1000, y=1000)
    params.orgPoint = mlib.Vector3f(x=0, y=-125, z=-20)
    params.xRange = mlib.Vector3f(x=150, y=150, z=0)
    params.yRange = mlib.Vector3f(x=0, y=150, z=150)

    dm = mlib.computeDistanceMapD(mp=mesh, params=params)
    mlib.saveDistanceMapToImage(distMap=dm, filename=tmp_path / "a.png")

    # === Verification
    map = mlib.loadDistanceMapFromImage(tmp_path / "a.png")
    aff = mlib.AffineXf3f()
    mesh = mlib.distanceMapToMesh(mp=map, toWorld=aff)
    mlib.saveMesh(mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh, skip_volume=True)


@pytest.mark.smoke
def test_distance_map_to_mesh(tmp_path):
    """
    Load distance from image, convert to mesh, compare with reference
    """
    #  Load input point
    case_name = f"dm-to-mesh"
    input_folder = Path(test_files_path) / "algorithms" / "distance_map"

    # load distance map

    dmap = mlib.loadDistanceMapFromImage(input_folder / "input_map.png")
    aff = mlib.AffineXf3f()
    mesh = mlib.distanceMapToMesh(mp=dmap, toWorld=aff)

    # === Verification
    mlib.saveMesh(mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh, skip_volume=True)
