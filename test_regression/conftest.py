from pathlib import Path
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-c-sharp-samples",
        action="store_true",
        default=False,
        help="Run tests for c# samples",
    )
    parser.addoption(
        "--csharp-sample-dir",
        action="store",
        help="Directory with c# sample",
        type=str,
        default=None
    )

@pytest.fixture
def csharp_sample_dir(request):
    if request.config.getoption("--csharp-sample-dir"):
        try:
            sample_exec_dir = Path(request.config.getoption("--csharp-sample-dir"))
        except Exception as e:
            print(e)
            raise Exception("Invalid --csharp-sample-dir")
    else:
        sample_exec_dir = Path("..\\source\\x64\\Release")
        print(f"WARNING: using default path for C# sample: {sample_exec_dir}")
    yield sample_exec_dir
