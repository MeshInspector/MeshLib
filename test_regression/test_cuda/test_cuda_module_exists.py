"""
Check that the `mrcudapy` module exists. Even on Mac, it should at least contain a stub `isCudaAvailable()` returning false.
"""

from meshlib import mrcudapy
import pytest
from module_helper import *


@pytest.mark.smoke
@pytest.mark.bindingsV3
def test_cuda_module_exists():
    mrcudapy.isCudaAvailable(); # Check that the function exists, but ignore the return value.
