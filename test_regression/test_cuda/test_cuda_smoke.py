import pytest


@pytest.mark.smoke
@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='negative'",
    reason="Only run when --run-cuda is 'negative'",
)
def test_cuda_not_available(cuda_module):
    with pytest.raises(ValueError):
        cuda_module.getRuntimeInfo()  # Should fail when no CUDA device is available


@pytest.mark.smoke
@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='positive'",
    reason="Only run when --run-cuda is 'positive'",
)
def test_cuda_available(cuda_module):
    info = cuda_module.getRuntimeInfo()  # Should succeed when a CUDA device is available
    assert info.fitForComputations(), "Check if cuda available and fit for computations"
