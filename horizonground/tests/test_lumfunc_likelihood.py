import pytest

import numpy as np
from horizonground.lumfunc_likelihood import (
    LumFuncLikelihood,
    LumFuncMeasurements,
)
from horizonground.lumfunc_modeller import quasar_PLE_lumfunc
from horizonground.utils import get_test_data_loc, load_parameter_set


@pytest.fixture(scope='module')
def measurements_file():
    return get_test_data_loc("eBOSS_QSO_LF_measurements.txt")


@pytest.fixture(scope='module')
def uncertainties_file():
    return get_test_data_loc("eBOSS_QSO_LF_uncertainties.txt")


@pytest.fixture(scope='module')
def prior_file():
    return get_test_data_loc("QSO_LF_PLE_model_prior.txt")


@pytest.fixture(scope='module')
def parameter_point(prior_file):
    return load_parameter_set(
        get_test_data_loc("QSO_LF_PLE_model_fixed.txt")
    ).values()


class TestLumFuncMeasurements:

    def test___init__(self, measurements_file, uncertainties_file):

        instance = LumFuncMeasurements(measurements_file, uncertainties_file)

        assert np.allclose(
            instance.luminosity_bins,
            [
                -28.8, -28.4, -28. , -27.6, -27.2, -26.8, -26.4, -26. , -25.6,
                -25.2, -24.8, -24.4, -24. , -23.6, -23.2, -22.8, -22.4, -22. ,
                -21.6, -21.2, -20.8
            ]
        )

        assert np.allclose(
            instance.redshift_bins,
            [0.87, 1.25, 1.63, 2.01, 2.4 , 2.8 , 3.25, 3.75]
        )

        assert instance.redshift_labels[0] == '$0.68<z<1.06$'

    def test___getitem__(self, measurements_file, uncertainties_file):

        instance = LumFuncMeasurements(measurements_file, uncertainties_file)

        measurements_in_bin, uncertainties_in_bin = instance['z=0.87']

        assert len(measurements_in_bin) == len(uncertainties_in_bin)

    def test_get_statistics(self, measurements_file, uncertainties_file):

        instance = LumFuncMeasurements(measurements_file, uncertainties_file)

        data_vector, variance_vector = instance.get_statistics()

        assert len(data_vector) == len(variance_vector)
        assert np.prod(np.isnan(data_vector)) == 0
        assert np.prod(np.isnan(variance_vector)) == 0


class TestLumFuncLumFuncLikelihood:

    def test___call__(self, parameter_point,
                      measurements_file, uncertainties_file,
                      prior_file):

        instance = LumFuncLikelihood(
            quasar_PLE_lumfunc,
            measurements_file,
            prior_file,
            uncertainty_file=uncertainties_file,
            fixed_file=None,
            prescription='poisson',
            model_options={'redshift_pivot': 2.22}
        )

        assert instance(parameter_point) == pytest.approx(-78.1366)
