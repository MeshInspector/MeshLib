import pytest

from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from pathlib import Path


@pytest.mark.smoke
def test_mesh_modification_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "mesh_modify"

    # === Code sample

    mesh = mrmeshpy.loadMesh(str(input_folder / "beethoven_in.stl"))
    # assert (expectedMesh.has_value())
    # mesh = expectedMesh.value()

    relaxParams = mrmeshpy.MeshRelaxParams()
    relaxParams.iterations = 5
    mrmeshpy.relax(mesh, relaxParams)

    props = mrmeshpy.SubdivideSettings()
    props.maxDeviationAfterFlip = 0.5
    mrmeshpy.subdivideMesh(mesh, props)

    plusZ = mrmeshpy.Vector3f()
    plusZ.z = 1.0
    rotationXf = mrmeshpy.AffineXf3f.linear(mrmeshpy.Matrix3f.rotation(plusZ, 3.1415 * 0.5))
    mesh.transform(rotationXf)

    # === Verification
    assert relative_hausdorff(mesh,
                              input_folder / "beethoven_out.stl") > DEFAULT_RHAUSDORF_THRESHOLD
