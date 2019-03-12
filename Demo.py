#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Demo.py: DEMONSTRATION
#
# Author: Mike Shengbo Wang
# Created: 2019-03-12
# =============================================================================


# ================================== LIBRARY ==================================


from nbodykit.source.catalog import CSVCatalog


# ================================= Execution =================================

file = CSVCatalog('./Data/BigMDPL_RockstarHalo_z1.0.txt',
                  ['mass', 'pid', 'x', 'y', 'z', 'vx', 'vy', 'vz']
                  )