"""
Check that the `mrcudapy` module exists. Even on Mac, it should at least contain a stub `getRuntimeInfo()` reporting that CUDA is not available.
"""

from meshlib import mrcudapy
import pytest
from module_helper import *


@pytest.mark.smoke
def test_cuda_module_exists():
    assert hasattr(mrcudapy, "getRuntimeInfo")  # Check that the function exists, without calling it (it raises when no CUDA device is available).
