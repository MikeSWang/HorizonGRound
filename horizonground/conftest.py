"""Test configuration for :mod:`horizonground`.

"""
import os
import sys
from pathlib import Path

import pytest

from .utils import load_parameter_set

current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))
test_data_dir = param_file_path = current_file_dir/"tests"/"test_data"


def pytest_addoption(parser):
    """Add command-line options to `pytest` parser.

    Parameters
    ----------
    parser : :class:`_pytest.config.argparsing.Parser`
        `pytest` parser object.

    """
    parser.addoption(
        '--runslow', action='store_true', default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Add ini-file options to `pytest` configuration.

    Parameters
    ----------
    config : :class:`_pytest.config.Config`
        `pytest` configuration object.

    """
    config.addinivalue_line('markers', "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Modify test collection items.

    Parameters
    ----------
    config : :class:`_pytest.config.Config`
        `pytest` configuration object.
    items : list of :class:`_pytest.nodes.Item`
        `pytest` item objects.

    """
    if config.getoption('--runslow'):
        return

    skip_slow = pytest.mark.skip(
        reason="use --runslow option to run slow tests"
    )
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def quasar_PLE_model_params(request):

    return load_parameter_set(
        test_data_dir/"eBOSS_QSO_LF_PLE_model_fits.txt"
    )


@pytest.fixture(scope="session", autouse=True)
def quasar_PLE_LEDE_model_params(request):

    return load_parameter_set(
        test_data_dir/"eBOSS_QSO_LF_PLE+LEDE_model_fits.txt"
    )


@pytest.fixture(scope="session", autouse=True)
def alpha_emitter_schechter_model_params(request):

    return load_parameter_set(
        test_data_dir/"H-alpha_LF_Schechter_model_fits.txt"
    )
