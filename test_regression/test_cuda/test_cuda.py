import pytest


@pytest.mark.smoke
@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='negative'",
    reason="Only run when --run-cuda is 'n'",
)
def test_cuda_not_available(cuda_module):
    assert cuda_module.isCudaAvailable() is False, "Check if cuda not available and it's reported correctly"


@pytest.mark.smoke
@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='positive'",
    reason="Only run when --run-cuda is 'p'",
)
def test_cuda_available(cuda_module):
    assert cuda_module.isCudaAvailable() is True, "Check if cuda available"
