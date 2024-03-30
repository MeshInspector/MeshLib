from module_helper import *
from pathlib import Path
from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff


def test_boolean_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "boolean"

    # === Sample code
    import meshlib.mrmeshpy as mrmeshpy

    # create first sphere with radius of 1 unit
    sphere1 = mrmeshpy.makeUVSphere(1.0, 64, 64)

    # create second sphere by cloning the first sphere and moving it in X direction
    sphere2 = mrmeshpy.copyMesh(sphere1)
    xf = mrmeshpy.AffineXf3f.translation(mrmeshpy.Vector3f(0.7, 0.0, 0.0))
    sphere2.transform(xf)

    # perform boolean operation
    result = mrmeshpy.boolean(sphere1, sphere2, mrmeshpy.BooleanOperation.Intersection)
    resultMesh = result.mesh
    if not result.valid():
        print(result.errorString)

    # save result to STL file
    mrmeshpy.saveMesh(resultMesh, str(tmp_path / "out_boolean.stl"))

    #  === Verification
    assert relative_hausdorff(tmp_path / "out_boolean.stl",
                              input_folder / "out_boolean.stl") > DEFAULT_RHAUSDORF_THRESHOLD
