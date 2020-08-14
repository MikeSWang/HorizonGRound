import numpy as np
import pytest

from horizonground.clustering_modification import (
    non_gaussianity_correction_factor,
    relativistic_correction_factor,
    standard_kaiser_factor,
)


REDSHIFT = 1.0
BIAS = 2.0
PNG = 1.0
WAVENUMBER = 1.e-3


@pytest.mark.parametrize(
    "ell,value",
    [
        (0, 5.31328), (2, 2.75664), (4, 0.17345)
    ]
)
def test_standard_kaiser_factor(ell, value):

    assert standard_kaiser_factor(ell, BIAS, REDSHIFT) == \
        pytest.approx(value, rel=1.e-3)


@pytest.mark.parametrize(
    "ell,value",
    [
        (0, 1.81196), (2, 0.42543), (4, 0.)
    ]
)
def test_non_gaussianity_correction_factor(ell, value):

    assert non_gaussianity_correction_factor(
        WAVENUMBER, ell, PNG, BIAS, REDSHIFT
    ) == pytest.approx(value, rel=1.e-3)


@pytest.mark.parametrize(
    "ell,value",
    [
        (0, 0.09541831), (2, 0.18921276), (4, 0.)
    ]
)
def test_relativistic_correction_factor(ell, value):

    assert np.isclose(
        relativistic_correction_factor(WAVENUMBER, ell, REDSHIFT, 0.), value,
        rtol=1.e-3
    )
