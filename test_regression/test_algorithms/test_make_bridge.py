from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mlib

import pytest


@pytest.mark.parametrize("input_case", ["torus", "crocodile"])
def test_make_bridge_edge(input_case, tmp_path):
    """
    Test make bridge edge algorithm
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "bridge" / input_case
    case_name = f"bridge_edge_{input_case}"
    mesh = mlib.loadMesh(input_folder / "input.mrmesh")
    topl = mesh.topology
    edges = topl.findHoleRepresentiveEdges()
    edges_num = topl.edgeSize()
    faces_num = topl.numValidFaces()
    mrmesh.makeBridgeEdge(topl, edges[0], edges[1])

    # === Verification
    with check:
        assert topl.edgeSize() - edges_num == 2, "Edges number should be increased on 2"
    with check:
        assert topl.numValidFaces() == faces_num, "Faces number should be the same"
    mlib.saveMesh(mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh)


@pytest.mark.parametrize("input_case", ["torus", "crocodile"])
def test_make_bridge(input_case, tmp_path):
    """
    Test make bridge edge algorithm
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "bridge" / input_case
    case_name = f"bridge_{input_case}"
    mesh = mlib.loadMesh(input_folder / "input.mrmesh")
    topl = mesh.topology
    edges = topl.findHoleRepresentiveEdges()
    edges_num = topl.edgeSize()
    faces_num = topl.numValidFaces()
    mrmesh.makeBridge(topl, edges[0], edges[1])

    # === Verification
    with check:
        assert mesh.topology.edgeSize() - edges_num == 6, "Edges number should be increased on 6"
    with check:
        assert topl.numValidFaces() - faces_num == 2, "Edges number should be increased on 2"

    mlib.saveMesh(mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh)
