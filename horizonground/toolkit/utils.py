# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# toolkit/utils.py: UTILITIES
#
# Copyright (C) 2019, MS Wang
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

""":mod:`~horizonground.toolkit.utils` provides various utilities.

"""
from glob import glob

import numpy as np


# =============================================================================
# DEFINITION
# =============================================================================

# FILE HANDLING TOOLS
# -----------------------------------------------------------------------------

def collate(filename_pattern, file_extension, headings=None, columns=None):
    """Collate data files.

    Parameters
    ----------
    filename_pattern : str
        String pattern of the file directory and name.
    file_extension : {'npy', 'txt', 'dat'}
        Data file extension.

    Returns
    -------
    collated_data : dict
        Collated data.
    count : int
        Number of data files collated.
    file : str
        Last collated file.

    Raises
    ------
    NotImplementedError
        If `file_extension` is not currently supported.
    ValueError
        If `file_extension` is ``'txt'`` or ``'dat'``, but `headings` or
        `columns` is `None`.
    ValueError
        If `headings` and `columns` are not in correpondence.

    Notes
    -----
    For text files, the data is assumed to be stored as a column-major 2-d
    array with each column in correspondence with a key in the returned
    :obj:`dict`.

    """
    if file_extension.lower()[-3:] == 'npy':
        data_all = []
        for file in glob(filename_pattern):
            data_all.append(np.load(file).item())

        count = len(data_all)

        # Initialise collated data using keys from the first data file.
        collated_data = dict.fromkeys(data_all[0].keys())
        for key in collated_data:
            collated_data[key] = np.concatenate(
                [np.atleast_1d(data[key]) for data in data_all], axis=0
                )

        return collated_data, count, file
    elif file_extension.lower()[-3:] in ['txt', 'dat']:
        # Consistency check.
        if headings is None or columns is None:
            raise ValueError(
                "`headings` or `columns` cannot be None "
                "when reading from non-'.npy' files. "
                )
        if len(headings) != len(columns):
            raise ValueError(
                "Lengths of `headings` and `columns` must agree. "
                )

        collated_data = {}
        for key in headings:
            collated_data.update({key: []})

        count = 0
        for file in glob(filename_pattern):
            data = np.loadtxt(file, usecols=columns)
            for keyidx, key in enumerate(headings):
                collated_data[key].append(np.atleast_1d(data[:,keyidx]))
            count += 1

        for key in headings:
            collated_data[key] = np.concatenate(collated_data[key], axis=0)

        return collated_data, count, file
    else:
        raise NotImplementedError("File extension currently unsupported. ")


# FORMATTING TOOLS
# -----------------------------------------------------------------------------

def float_format(x, case):
    r"""Format float as a string.

    Parameters
    ----------
    x : float
        Number to be formatted.
    case : {'latex', 'sci', 'intdot', 'decdot'}
        Format case, one of :math:`\text{\LaTeX}` (``'latex'``), scientific
        (``'sci'``), rounded integer with a decimal dot (``'intdot'``), or
        a float *whose first decimal is 0* represented as a rounded integer
        with a decimal dot (``'decdot'``).

    Returns
    -------
    x_str : str
        Formatted string.

    """
    if not isinstance(x, float):
        x = float(x)

    if case.lower() == 'latex':
        x_str = "{:g}".format(x)
        if "e" in x_str:
            base, exponent = x_str.split("e")
            x_str = r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    elif case.lower() == 'sci':
        x_str = "{:g}".format(x)
        if "e" in x_str:
            x_str = x_str.replace("e+0", "e+").replace("e-0", "e-")
    elif case.lower() == 'intdot':
        x_str = "{}".format(np.around(x)).strip("0")
    elif case.lower() == 'decdot':
        x_to1dp = "{:.1f}".format(x)
        if x_to1dp[-1] == '0':
            x_str = x_to1dp.strip("0")
        else:
            x_str = x_to1dp

    return x_str
