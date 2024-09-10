import pytest

from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from pathlib import Path


@pytest.mark.smoke
def test_grid_traingulation_samlpe(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "grid_triangulation"

    # === Sample code
    import numpy as np

    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    mesh = mrmeshnumpy.meshFromUVPoints(x, y, z)

    #  === Verification
    assert relative_hausdorff(mesh,
                              input_folder / "trng_grid.stl") > DEFAULT_RHAUSDORF_THRESHOLD
