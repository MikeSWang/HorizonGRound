#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *****************************************************************************
# toolkit/utils.py: UTILITY TOOLS
#
# Author: MS Wang
# Created: 2019-03
# *****************************************************************************

""":mod:`toolkit.utils`---user-customised utilities"""

# =============================================================================
# LIBRARY
# =============================================================================

from glob import glob

import numpy as np


# =============================================================================
# DEFINITION
# =============================================================================

# FILE HANDLING TOOLS
# -----------------------------------------------------------------------------

def collate(filename_pattern, file_datatype):
    """Collate saved data output files into a single file.

    Parameters
    ----------
    filename_pattern : str
        String of the file directory and name
    file_datatype : str
        Data type inside saved files

    Returns
    -------
    result : :type:`file_datatype`
        Collated result in the data type of ``file_datatype``
    count : int
        Number of data output files collated
    file : str
        Last accessed file name

    TODO: Implement other file data types than :type:`dict`.

    Raises
    ------
    NotImplementedError
        If ``file_datatype`` is not :type:`dict`
    """

    if file_datatype == 'dict':
        # Reading and counting
        measurements = []
        for file in glob(filename_pattern):
            measurements.append(np.load(file).item())
        count = len(measurements)

        # Collating
        result = dict.fromkeys(measurements[0].keys())
        for key in result:
            result[key] = np.concatenate([m[key] for m in measurements],
                                         axis=0
                                         )

        return result, count, file
    else:
        raise NotImplementedError


# FORMATTING TOOLS
# -----------------------------------------------------------------------------

def latex_float(x):
    r"""Format number in \LaTeX string with scientific notation.

    Parameters
    ----------
    x : float
        Number to be formatted

    Returns
    -------
    x_str : str
        Formatted string
    """

    x_str = "{0:.1g}".format(x)
    if "e" in x_str:
        base, exponent = x_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return x_str


# TRANSFORMATION TOOLS
# -----------------------------------------------------------------------------

def fftconv(fval_nd, gval_nd):
    """FFT convolution of n-d function values.

    Parameters
    ----------
    fval_nd, gval_nd : float, array_like
        Functions to be convolved in n-d using FFT

    Returns
    -------
    fconvg_val_nd : float, array_like
        Convolved real values in n-d
    """

    from scipy.fftpack import fftn, ifftn

    fconvg_val_nd = ifftn(fftn(fval_nd) * fftn(gval_nd)).real

    return fconvg_val_nd
