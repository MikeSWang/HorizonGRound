"""Utilities tools.

String formatting
-----------------

.. autosummary::

    process_header


"""

__all__ = [
    'process_header',
]


def process_header(header, skipcols=0):

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
