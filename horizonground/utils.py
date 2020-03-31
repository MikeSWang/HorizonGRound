"""
Utilities (:mod:`~horizonground.utils`)
===========================================================================

Utilities tools.


String formatting
-----------------

.. autosummary::

    process_header

|

"""

__all__ = [
    'process_header',
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
    headings : list of str
        Headings of the file.

    """
    header = header.strip("#").strip("\n")

    if "," in header:
        headings = list(
            map(
                lambda heading: heading.strip(),
                header.split(",")[skipcols:]
            )
        )
    else:
        headings = [
            heading.strip() for heading in header.split()[skipcols:]
            if not heading.isspace()
        ]

    return headings
