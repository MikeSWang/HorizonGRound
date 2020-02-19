"""Configuration for scripts.

"""
import os
import sys
from pathlib import Path

import matplotlib as mpl
import seaborn as sns


def use_local_package(package_paths):
    """Add local package path(s) to `sys.path` for Python module search.

    Parameters
    ----------
    package_paths : (list of) str
        Local package paths.

    Examples
    --------
    >>> use_local_package("mypkg/"); "mypkg" in sys.path[0]
    True

    """
    if isinstance(package_paths, list):
        package_paths = [
            os.path.abspath(relpath) for relpath in package_paths
        ]
        sys.path = package_paths + sys.path
    else:
        package_paths = os.path.abspath(package_paths)
        sys.path.insert(0, package_paths)


def major_version(package):
    """Return the major version of a package.

    Parameters
    ----------
    package : module
        Package.

    Return
    ------
    int
        Major version.

    """
    return int(package.__version__.split(".")[0])


def sci_notation(num):
    """Format integers in scientific notation.

    Parameters
    ----------
    num : int
        Integer to be formatted.

    Returns
    -------
    num_str : str
        Formatted string in scientific notation.

    """
    base, index = "{:.1e}".format(num).split("e")
    base = base.rstrip(".0").replace(".", ",")
    index = index.lstrip("+").lstrip("-").lstrip("0")

    num_str = "E".join([base, index])

    return num_str

mpl.pyplot.style.use(
    mpl.rc_params_from_file(
        "../config/horizon.mplstyle",
        use_default_template=False
    )
)

sns.set(style='ticks', font='serif')

PATH = Path("../data/")
PATHEXT = Path("../data/external")
PATHIN = Path("../data/input")
PATHOUT = Path("../data/output")
