import pytest
from lifeguide.config import settings_from_env

def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", default=False,
                     help="run integration tests (needs docker PG)")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="need --integration and docker PG")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

@pytest.fixture
def settings():
    return settings_from_env()
