"""
###########################################################################
``HorizonGRound``
###########################################################################

.. topic:: Licence Statement

    Copyright (C) 2020, M S Wang

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program.  If not, see `<https://www.gnu.org/licenses/>`_.

"""
with open("../version.txt", 'r') as version_info:
    version_tag, version = [v.strip() for v in version_info]
    if version_tag == 'latest':
        branch = 'master'
    else:
        branch = version_tag

__author__ = "Mike S Wang"
__contact__ = "Mike S Wang"
__copyright__ = "Copyright 2020, M S Wang"
__date__ = "2020/07/01"
__description__ = (
    "Forward-modelling of relativistic effects in ultra-large-scale "
    "clustering from the tracer luminosity function."
)
__email__ = "mike.wang@port.ac.uk"
__license__ = "GPLv3"
__version__ = version
__url__ = "https://github.com/MikeSWang/HorizonGRound/tree/{}".format(branch)
