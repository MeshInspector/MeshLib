def pytest_addoption(parser):
    parser.addoption(
        "--run-c-sharp-samples",
        action="store_true",
        default=False,
        help="Run tests for c# samples",
    )