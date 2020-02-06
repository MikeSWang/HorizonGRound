r"""
Luminosity Function Likelihood (:mod:`~horizonground.lumfunc_likelihood`)
===========================================================================

Fit tracer luminosity functions and produce model constraints.

Data processing
---------------

.. autosummary::

    LumFuncMeasurements


Likelihoods
-----------

.. autosummary::

    normal_log_pdf
    LumFuncLikelihood

|

"""
from __future__ import division

from itertools import compress as iterfilter
from itertools import product as iterproduct

import numpy as np


# Data processing
# -----------------------------------------------------------------------------

class LumFuncMeasurements:
    """Luminosity function measurements for a tracer sample.

    Parameters
    ----------
    data_file : str or :class:`pathlib.Path`
        Luminosity function data file path.
    base10_log : bool, optional
        If `True` (default), all values are converted to base-10
        logarithms.

    Attributes
    ----------
    redshift_labels : list of str
        Redshift bin labels.
    redshift_bins : list of float
        Redshift bin centres.
    brightness_bins : list of float
        Luminosity/magnitude bin centres.

    Notes
    -----
    The data file is assumed to be an array where rows correspond to
    increasing luminosity/magnitude bins (except the first line for
    headings) and alternating columns are base-10 logarithmic luminosity
    function values and uncertainties for increasing redshift bins (except
    the first column for luminosity/magnitude bin centres).

    """

    def __init__(self, data_file, base10_log=True):

        self._source_path = data_file

        self._load_data_file(self._source_path, base10_log)

        self._point_validity = ~np.isnan(self._measurements.flatten()) \
            & ~np.isnan(self._uncertainties.flatten())

    def get_statistics(self):
        """Return the empirical mean and covariance from luminosity
        function data.

        Returns
        -------
        data_mean : :class:`numpy.ndarray`
            Empirical mean.
        data_var : :class:`numpy.ndarray`
            Empirical variance as a diagonal matrix.

        Notes
        -----
        Data vectors are ordered by increasing redshift bins and
        for the same redshift bin ordered by increasing
        luminosity/magnitude.

        """
        data_mean = self._measurements.flatten()[self._point_validity]
        data_var = np.diag(self._uncertainties.flatten()[self._point_validity])

        return data_mean, data_var

    def _load_data_file(self, data_file, base10_log):

        raw_data = np.genfromtxt(data_file, unpack=True)

        with open(data_file, 'r') as file:
            headings = list(
                map(
                    lambda header: header.strip(" "),
                    file.readline().strip("#").strip("\n").split(",")[1:]
                )
            )

        measurement_array = raw_data[1::2]
        uncertainty_array = raw_data[2::2]

        if not base10_log:
            for z_idx, (var_name, d_var_name) \
                    in enumerate(zip(headings[0::2], headings[1::2])):
                if 'lg_' in var_name:
                    measurement_array[z_idx] = 10**measurement_array[z_idx]
                if 'lg_' in d_var_name:
                    uncertainty_array[z_idx] = measurement_array[z_idx] \
                        * (10**uncertainty_array[z_idx] - 1)
            headings = [head.replace("lg_", "") for head in headings]

        self._measurements = measurement_array
        self._uncertainties = uncertainty_array
        self.redshift_labels, self.redshift_bins = \
            self._extract_redshift_bins(headings)
        self.brightness_bins = raw_data[0]

    @staticmethod
    def _extract_redshift_bins(data_names):

        bin_labels = sorted(list(set(
            map(
                r"${}$".format,
                [
                    dname.split("z_")[-1].replace("_", "<z<")
                    for dname in data_names if 'z_' in dname
                ]
            )
        )))

        bin_centres = [
            np.mean(tuple(map(float, blabel.strip("$").split("<z<"))))
            for blabel in bin_labels
        ]

        return bin_labels, bin_centres

    def __getitem__(self, z_key):
        """Get luminosity function measurements and uncertainties
        for a specific redshift bin.

        Parameters
        ----------
        z_key: int, slice or str
            Slice or integer index or string representing a redshift
            bin.  If a string, the accepted format is e.g. ``z=1.0``.

        Returns
        -------
        :class:`numpy.ndarray`, :class:`numpy.ndarray`
            Measurements and uncertainties for the redshift bin.

        """
        if isinstance(z_key, int) or isinstance(z_key, slice):
            z_idx = z_key
        else:
            try:
                z = float(str(z_key).replace(" ", "").lstrip("z="))
                z_idx = self.redshift_bins.index(z)
            except (TypeError, ValueError):
                raise KeyError(
                    "No measurements for redshift bin '{}'. ".format(z_key)
                )

        return self._measurements[z_idx], self._uncertainties[z_idx]

    def __str__(self):

        return (
            "LuminosityFunctionMeasurements(data_source='{}')"
            .format(self._source_path)
        )


# Likelihood construction
# -----------------------------------------------------------------------------

def normal_log_pdf(data_vector, model_vector, covariance_matrix):
    """Compute the logarithmic probability density for normal
    distributions.

    Parameters
    ----------
    data_vector : array_like
        Data vector.
    model_vector : array_like
        Model vector.
    covariance_matrix : array_like
        Covariance matrix.

    Returns
    -------
    log_p : :class:`numpy.ndarray`
        Log probability density.

    """
    data_vector = np.asarray(data_vector)
    model_vector = np.asarray(model_vector)

    log_p = - 1/2 * np.matmul(
        (data_vector - model_vector).T,
        np.matmul(covariance_matrix, data_vector - model_vector)
    )

    return log_p


class LumFuncLikelihood(LumFuncMeasurements):
    """Luminosity function likelihood.

    Notes
    -----
    The built-in likelihood distribution is multivariate normal and
    assumes a diagonal covariance matrix estimate without any
    corrections for its uncertainties.

    Parameters
    ----------
    lumfunc_model : callable
        Luminosity function model.
    prior_file : str or :class:`pathlib.Path`
        Luminosity function model prior file path.
    data_file : str or :class:`pathlib.Path`
        Luminosity function data file path.
    base10_log : bool, optional
        If `True` (default), all values are converted to base-10
        logarithms.

    """

    def __init__(self, lumfunc_model, prior_file, data_file, base10_log=True):

        super().__init__(data_file, base10_log=base10_log)

        self._lg_conversion = base10_log
        self._lumfunc_model = lumfunc_model

        self._priors = self._setup_prior(prior_file)

        self._data_points = list(
            iterfilter(
                map(
                    lambda tup: tuple(reversed(tup)),
                    iterproduct(self.redshift_bins, self.brightness_bins)
                ),
                self._point_validity
            )
        )

        self._data_vector, self._data_covariance = self.get_statistics()

    def __call__(self, **model_params):
        """Return log-likelihood value at input model parameter point.

        Parameters
        ----------
        **model_params
            All model parameters associated with :attr:`lumfunc_model`
            except the luminosity function arguments
            (luminosity/magnitude and redshift).

        Returns
        -------
        float
            Log-likelihood value.

        """
        for param_name, param_val in model_params.items():
            if param_val < self._priors[param_name][0] \
                    or param_val > self._priors[param_name][-1]:
                return - np.inf

        if self._lg_conversion:
            model_vector = [
                np.log10(self._lumfunc_model(*data_point, **model_params))
                for data_point in self._data_points
            ]
        else:
            model_vector = [
                self._lumfunc_model(*data_point, **model_params)
                for data_point in self._data_points
            ]

        return normal_log_pdf(
            self._data_vector, model_vector, self._data_covariance
        )

    def _setup_prior(self, parameter_prior_file):

        with open(parameter_prior_file, 'r') as file:
            parameter_names = list(
                map(
                    lambda header: header.strip(" "),
                    file.readline().strip("#").strip("\n").split(",")
                )
            )

        prior_data = np.genfromtxt(parameter_prior_file, unpack=True)
        prior_ranges = [prior_ends for prior_ends in prior_data]

        return dict(zip(parameter_names, prior_ranges))

    def __str__(self):

        return (
            "LuminosityFunctionLikelihood"
            "(data_source='{}',lum_func_model='{}',prob_model='{}')"
            .format(
                self._source_path,
                self._lumfunc_model.__name__,
                normal_log_pdf.__name__)
        )
