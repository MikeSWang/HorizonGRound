"""
Utilities (:mod:`~horizonground.utils`)
===========================================================================

Utilities tools.

.. autosummary::

    process_header
    load_parameter_set

|

"""

__all__ = [
    'process_header',
    'load_parameter_set',
]


def process_header(header, skipcols=0):
    r"""Process comment-line header (indicated by the \'#\' character) of
    text files.

    Parameters
    ----------
    header : str
        File header line.
    skipcols : int, optional
        Skip the first columns (default is 0).

    Returns
    -------
    columns : list of str
        Column headings of the file.

    """
    header = header.strip("#").strip("\n")

    if "," in header:
        columns = list(map(
            lambda heading: heading.strip(), header.split(",")[skipcols:]
        ))
    else:
        columns = [
            heading.strip()
            for heading in header.split()[skipcols:] if not heading.isspace()
        ]

    return columns


def load_parameter_set(parameter_file):
    """Load a parameter set from a file into a dictionary.

    Parameters
    ----------
    parameter_file : *str or* :class:`pathlib.Path`
        Parameter file.

    Returns
    -------
    parameter_set : dict
        Parameter set.

    """
    with open(parameter_file, 'r') as pfile:
        parameters = process_header(pfile.readline())
        estimates = tuple(map(float, pfile.readline().split(",")))

    parameter_set = dict(zip(parameters, estimates))
    for parameter in parameters:
        if parameter.startswith(r"\Delta"):
            del parameter_set[parameter]

    return parameter_set
