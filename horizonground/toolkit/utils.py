#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *****************************************************************************
# toolkit/utils.py: UTILITY TOOLS
#
# Author: MS Wang
# Created: 2019-03
# *****************************************************************************

""":mod:`toolkit.utils` provides user-customised utilities.

"""

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

def collate(filename_pattern, file_datatype, headings=None, columns=None):
    """Collate saved data output files into a single file.

    Parameters
    ----------
    filename_pattern : str
        String of the file directory and name.
    file_datatype : str
        Data type inside saved files.

    Returns
    -------
    result
        Collated result in the data type of ``file_datatype``.
    count : int
        Number of data output files collated.
    file : str
        Last accessed file name.

    Raises
    ------
    NotImplementedError
        If `file_datatype` is not :data:`dict` or ``.txt``.
    ValueError
        If `file_datatype` is ``.txt`` but `headings` or `columns` is `None`.
    ValueError
        If `headings` and `columns` are not in correpondence.

    TODO: Implement other file data types than :data:`dict` and ``.txt``

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
    elif file_datatype == 'txt':
        # Consistency check
        if headings is None or columns is None:
            raise ValueError("`headings` or `columns` cannot be `None` "
                             "when reading from text files. "
                             )
        if len(headings) != len(columns):
            raise ValueError("Numbers of elements in `headings` and `columns` "
                             "must agree. "
                             )

        # Collating
        result = {}
        for key in headings:
            result.update({key: []})

        count = 0
        for file in glob(filename_pattern):
            count += 1
            ndarray_file = np.loadtxt(file, usecols=columns)
            for keyidx, key in enumerate(headings):
                result[key].append((ndarray_file[:,keyidx])[None,:])

        for key in headings:
            result[key] = np.concatenate(result[key], axis=0)

        return result, count, file
    else:
        raise NotImplementedError("Data type currently unsupported. ")


# FORMATTING TOOLS
# -----------------------------------------------------------------------------

def float_format(x, case):
    """Format float as a string.

    Parameters
    ----------
    x : float
        Number to be formatted.
    case : {'latex', 'sci'}
        Format case.

    Returns
    -------
    x_str : str
        Formatted string.

    """

    if case == 'latex':
        x_str = "{:.1g}".format(x)
        if "e" in x_str:
            base, exponent = x_str.split("e")
            x_str = r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    elif case == 'sci':
        x_str = "{:.0e}".format(x).replace("e-0", "e-")
    elif case == 'intdec':
        x_str = "{:.1f}".format(x).strip("0")
    elif case == 'rounddec':
        x_str = "{}".format(np.around(x)).strip("0")

    return x_str


# TRANSFORMATION TOOLS
# -----------------------------------------------------------------------------

def fftconv(fval_nd, gval_nd):
    """FFT convolution of n-d function values.

    Parameters
    ----------
    fval_nd, gval_nd : float, array_like
        Functions to be convolved in n-d using FFT.

    Returns
    -------
    fconvg_val_nd : float, array_like
        Convolved real values in n-d.
        
    """

    from scipy.fftpack import fftn, ifftn

    fconvg_val_nd = ifftn(fftn(fval_nd) * fftn(gval_nd)).real

    return fconvg_val_nd
