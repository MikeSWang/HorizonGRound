"""Utilities tools.

String formatting
-----------------

.. autosummary::

    process_header


"""

__all__ = [
    'process_header',
]


def process_header(header):

    header = header.strip("#").strip("\n")

    if "," in header:
        headings = list(
            map(
                lambda heading: heading.strip(),
                header.split(",")[1:]
            )
        )
    else:
        headings = [
            heading.strip() for heading in header.split()[1:]
            if not heading.isspace()
        ]

    return headings
