import math
import meshlib.mrmeshpy as mr
import pytest

from helpers.meshlib_helpers import compare_meshes_similarity
from module_helper import *
from pytest_check import check
from constants import test_files_path
from pathlib import Path

@pytest.mark.smoke
@pytest.mark.bindingsV3
def test_issue_2899(tmp_path):
    """
    Test for issue 2899 in MeshLib
    https://github.com/MeshInspector/MeshLib/issues/2899#issuecomment-2203088307

    This is a specific code that solved issue of customer: to perform slope and slide on mesh with holes and
    non-manifold vertices
    """
    input_folder = Path(test_files_path) / "issues" / "2899"

    # Open mesh files
    slide = mr.loadMesh(input_folder / "slide_clip.off")
    slope = mr.loadMesh(input_folder / "slope_clip.off")

    # Put input meshes in one as separate connected components
    mesh = slope
    mesh.addMesh(slide)

    # Find cut contour as the edges having the same coordinates in both Slide and Slope
    twin_pairs = mr.findTwinEdgePairs(mesh, 0)
    twin_edges = mr.findTwinUndirectedEdges(twin_pairs)
    twin_map = mr.findTwinUndirectedEdgeHashMap(twin_pairs)

    # Eliminate edges shorter than 10
    settings = mr.DecimateSettings()
    settings.strategy = mr.DecimateStrategy.ShortestEdgeFirst
    settings.maxError = 10

    # Collapse contour edges together in Slide and Slope
    settings.twinMap = twin_map
    # And never flip them
    settings.notFlippable = twin_edges
    # Or move due to collapse of nearby edges
    bd_verts = mr.getIncidentVerts(mesh.topology, twin_edges)
    settings.bdVerts = bd_verts
    settings.touchBdVerts = False
    settings.collapseNearNotFlippable = True
    # Allow flipping of other edges
    settings.maxAngleChange = math.pi / 3

    # Process edges not further than 3 hops away from cut contour
    region_edges = mr.getIncidentEdges(
        mesh.topology,
        mr.getIncidentEdges(
            mesh.topology,
            mr.getIncidentEdges(mesh.topology, twin_edges)
        )
    )
    settings.edgesToCollapse = region_edges

    # Run re-meshing
    mr.decimateMesh(mesh, settings)

    # Separate result on Slide and Slope
    slope_faces = mr.MeshComponents.getLargestComponent(mesh)
    slope_res = mr.Mesh()
    slope_res.addMeshPart(mr.MeshPart(mesh, slope_faces))
    slide_res = mr.Mesh()
    slide_res.addMeshPart(mr.MeshPart(mesh, mesh.topology.getValidFaces() - slope_faces))

    # Save in files
    mr.saveMesh(slope_res, tmp_path / "slopeRes.off")
    mr.saveMesh(slide_res, tmp_path / "slideRes.off")

    # Check results
    slope_res_ref = mr.loadMesh(input_folder / "slopeRes_ref.off")
    slide_res_ref = mr.loadMesh(input_folder / "slideRes_ref.off")
    with check:
        compare_meshes_similarity(slope_res, slope_res_ref)
    with check:
        compare_meshes_similarity(slide_res, slide_res_ref)
