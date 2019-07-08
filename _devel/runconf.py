#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# runconf.py: DEVELOPMENT SCRIPT CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Runtime configuration overheads for example scripts."""

from sys import argv, path
from os.path import basename, splitext

path.append("../")

from horizonground.studio.style import horizon_style as hgrstyle

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def filename(*filepath):
    if not filepath:
        return splitext(basename(argv[0]))[0]
    else:
        return splitext(basename(filepath[0]))[0]
