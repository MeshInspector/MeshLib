import pytest

from meshlib import mrcudapy as mc


@pytest.mark.smoke
@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='negative'",
    reason="Only run when --run-cuda is 'n'",
)
def test_cuda_not_available():
    assert mc.isCudaAvailable() is False, "Check if cuda not available and it's reported correctly"


@pytest.mark.smoke
@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='positive'",
    reason="Only run when --run-cuda is 'p'",
)
def test_cuda_available():
    assert mc.isCudaAvailable() is True, "Check if cuda available"
