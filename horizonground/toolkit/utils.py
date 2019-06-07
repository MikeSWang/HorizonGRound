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

from functools import lru_cache
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


# PROCESSING TOOLS
# -----------------------------------------------------------------------------

def allocate_task(ntask, nproc):
    """Allocate tasks to cores.

    Parameters
    ----------
    ntask : int
        Number of tasks
    nproc : int
        Number of processors

    Returns
    -------
    tasks : list of int
        Number of tasks for each processor
    """

    ntask_remain, nproc_remain, tasks = ntask, nproc, []
    while ntask_remain > 0:
        ntask_assign = ntask_remain // nproc_remain
        tasks = np.append(ntask_assign)
        ntask_remain -= ntask_assign
        nproc_remain -= 1

    return tasks


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


# ALGEBRAIC ALGORITHMS
# -----------------------------------------------------------------------------

def scan_interval(func, a, b, dx):
    """Scan interval from lower end to detect sign change.

    Parameters
    ----------
    func : function
        Function whose sing-change interval is to be found
    a, b: float
        Starting interval end points
    dx : float
        Increment for lower end point

    Returns
    -------
    x0, x1 : float or None
        End points for a bracket with sign change (``None``, ``None`` if scan
        unsuccessful)
    """

    # Starting interval of increment width
    x0, x1 = a, a + dx
    f0, f1 = func(x0), func(x1)

    # No change detected yet
    while f0 * f1 >= 0:
        # Terminate when full interval scanned
        if x0 >= b:
            return None, None
        # Incrementally move interval
        x0, x1 = x1, x1 + dx
        f0, f1 = f1, func(x1)

    return x0, x1


def bisect(func, x0, x1, epscvg=1.0e-9):
    """Bisection method for root finding.

    Parameters
    ----------
    func : function
        Function whose zero bracket is to be found
    x0, x1: float
        Starting interval end points
    eps : float
        Precision control for convergence (through maximum iteration)

    Returns
    -------
    A single possible root (float or NoneType).
    """

    # Basic checks
    f0, f1 = func(x0), func(x1)
    if f0 == 0:
        return x0
    elif f1 == 0:
        return x1
    elif f0 * f1 > 0:
        print("Root is not bracketed.")
        return None

    # Determine maximum iteration given precision number
    niter = int(np.ceil(np.log(np.abs(x1 - x0)/epscvg) /
                        np.log(2.0)
                        )
                )

    # Iterative bisection
    for i in range(niter):
        # Mid-point
        x2 = (x0 + x1) / 2
        f2 = func(x2)
        # Root found
        if f2 == 0:
            return x2
        # Sign change detected, move lower end point
        if f1 * f2 < 0:
            x0 = x2
            f0 = f2
        # Sign change not detected, move upper end point
        else:
            x1 = x2
            f1 = f2

    return (x0 + x1) / 2


@lru_cache()
def find_roots(func, a, b, maxnum=np.inf, eps=1e-5):
    """Find all roots of a function in an interval.

    Parameters
    ----------
    func : function
        Function whose zeros are to be found
    a, b : float
        Interval end points
    maxnum : int
        Maximum number of roots needed from below (default is ``np.inf``)
    eps : float
        Precision required (default is 1e-5)

    Returns
    -------
    roots : float, array_like
        Possible roots
    """

    roots = []
    while len(roots) < maxnum:
        x0, x1 = scan_interval(func, a, b, eps)
        # Sign change interval found
        if x0 != None:
            # Bisect interval to find root
            root = bisect(func, x0, x1)
            # Round non-empty result
            if root != None:
                roots.append(round(root, -int(np.log10(eps))))
            # Reset interval for next root
            a = x1
        # Sign change interval not found in remaining range, return & terminate
        else:
            return np.asarray(roots)
            break

    return np.asarray(roots, dtype=float)


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
