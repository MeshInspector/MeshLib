import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-cuda",
        action="store",
        help="Run cuda tests: positive, negative or skip",
        type=str,
        default='skip',
        choices=("positive", "negative", "skip")
    )

@pytest.fixture(scope="module")
def cuda_module():
    from meshlib import mrcudapy as mc
    return mc
