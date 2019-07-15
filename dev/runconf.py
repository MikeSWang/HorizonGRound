# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# runconf.py: RUNTIME CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Runtime configuration for scripts."""

from sys import argv, path
from os.path import basename, splitext

path.append("../")  # for the following imports

from horizonground.studio import hgrstyle
from horizonground.toolbox import float_format as ff

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def filename(*filepath):
    if not filepath:
        return splitext(basename(argv[0]))[0]
    else:
        return splitext(basename(filepath[0]))[0]
