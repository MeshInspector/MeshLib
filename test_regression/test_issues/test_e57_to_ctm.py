from pathlib import Path

import meshlib.mrmeshpy as mm
import pytest
from constants import test_files_path
from helpers.meshlib_helpers import compare_points_similarity


@pytest.mark.bindingsV3
@pytest.mark.timeout(5)
def test_e57_to_ctm(tmp_path):
    """
    Test checks hang while saving color points into ctm.
    https://github.com/MeshInspector/MeshInspectorCode/issues/5534

    Test marked with timeout, but it won't work on my machine with this hang.
    If this test will hang in pipeline - test failed, the issue is returned.
    """
    import random
    input_folder = Path(test_files_path) / "issues" / "5534"

    # Generating input
    points = mm.VertCoords()
    colors = mm.VertColors()

    num_points = 100
    for _ in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        z = random.uniform(0, 1)
        points.vec.append(mm.Vector3f(x, y, z))

        r = random.uniform(0, 1)
        g = random.uniform(0, 1)
        b = random.uniform(0, 1)
        colors.vec_.append(mm.Color(r, g, b))

    cloud = mm.PointCloud()
    cloud.points = points
    points_set = mm.VertBitSet()
    points_set.resize(num_points, True)
    cloud.validPoints = points_set

    save_settings = mm.SaveSettings()
    save_settings.colors = colors

    save_settings.progress = lambda x: True

    mm.PointsSave.toCtm(cloud, tmp_path / "e57_out.ctm", save_settings)
    compare_points_similarity(input_folder / "e57_out.ctm", tmp_path / "e57_out.ctm", rhsdr_thresh=0.1)
    # skipping hausdorff verification because of randomization
