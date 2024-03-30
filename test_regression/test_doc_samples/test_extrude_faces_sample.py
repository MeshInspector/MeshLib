from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from pathlib import Path


def test_extrude_faces_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "extrude_faces"

    # === Sample code
    import meshlib.mrmeshpy as mrmeshpy

    # Load mesh
    mesh = mrmeshpy.loadMesh(str(input_folder / "fox_geometrik.stl"))

    # Prepare region to extrude
    facesToExtrude = mrmeshpy.FaceBitSet()
    facesToExtrude.resize(3, False)
    facesToExtrude.set(mrmeshpy.FaceId(1), True)
    facesToExtrude.set(mrmeshpy.FaceId(2), True)

    # Create duplicated verts on region boundary
    mrmeshpy.makeDegenerateBandAroundRegion(mesh, facesToExtrude)

    # Find vertices that will be moved
    vertsForMove = mrmeshpy.getIncidentVerts(mesh.topology, facesToExtrude)

    # Move each vertex
    for v in range(vertsForMove.size()):
        if vertsForMove.test(mrmeshpy.VertId(v)):
            mesh.points.vec[v] += mrmeshpy.Vector3f(0.0, 0.0, 1.0)

    # Invalidate internal caches after manual changing
    mesh.invalidateCaches()

    # Save mesh
    mrmeshpy.saveMesh(mesh, str(tmp_path / "extrudedMesh.stl"))

    #  === Verification
    assert relative_hausdorff(input_folder / "extrudedMesh.stl",
                              tmp_path / "extrudedMesh.stl") > DEFAULT_RHAUSDORF_THRESHOLD
