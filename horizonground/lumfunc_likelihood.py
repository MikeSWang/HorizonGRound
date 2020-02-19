r"""
Luminosity Function Likelihood (:mod:`~horizonground.lumfunc_likelihood`)
===========================================================================

Fit tracer luminosity functions and produce model constraints.


Data processing
---------------

.. autosummary::

    LumFuncMeasurements


Likelihood evaluation
---------------------

.. autosummary::

    LumFuncLikelihood

|

"""
from __future__ import division

from collections import OrderedDict
from itertools import compress as iterfilter
from itertools import product as iterproduct

import numpy as np


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

        self._lg_conversion = base10_log
        self._data_source_path = data_file
        self._load_data_file()

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

    def _load_data_file(self):

        raw_data = np.genfromtxt(self._data_source_path, unpack=True)

        with open(self._data_source_path, 'r') as file:
            headings = list(
                map(
                    lambda header: header.strip(" "),
                    file.readline().strip("#").strip("\n").split(",")[1:]
                )
            )

        measurement_array = raw_data[1::2]
        uncertainty_array = raw_data[2::2]

        if not self._lg_conversion:
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

    def __str__(self):

        return (
            "LuminosityFunctionMeasurements(data_source='{}')"
            .format(self._data_source_path)
        )


def _uniform_log_pdf(param_vals, param_ranges):

    if isinstance(param_vals, np.ndarray):
        param_vals = param_vals.tolist()

    for p_val, p_range in zip(param_vals, list(param_ranges)):
        if p_val < p_range[0] or p_val > p_range[-1]:
            return - np.inf

    return 0


def _normal_log_pdf(data_vector, model_vector, covariance_matrix):

    data_vector = np.asarray(data_vector)
    model_vector = np.asarray(model_vector)

    if not all(np.isfinite(model_vector)):
        return - np.inf

    log_p = - 1/2 * np.linalg.multi_dot(
        [
            data_vector - model_vector,
            np.linalg.inv(covariance_matrix),
            data_vector - model_vector
        ]
    )

    return log_p


class LumFuncLikelihood(LumFuncMeasurements):
    """Luminosity function likelihood.

    Notes
    -----
    The built-in likelihood distribution is multivariate normal and
    assumes a diagonal covariance matrix estimate without any
    corrections for its uncertainties.  The built-in prior distribution
    is multivariate uniform.

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
    fixed_file : str or :class:`pathlib.Path` or None, optional
        Luminosity function model fixed parameter file path.

    """

    def __init__(self, lumfunc_model, prior_file, data_file, base10_log=True,
                 fixed_file=None):

        self._lumfunc_model = lumfunc_model
        self._prior_source_path = prior_file
        self._fixed_source_path = fixed_file

        super().__init__(data_file, base10_log=base10_log)

        self.data_points = self._setup_data_points()
        self.prior, self.fixed = self._setup_prior()

        self._data_vector, self._data_covariance = self.get_statistics()

    def __call__(self, param_point, use_prior=False):
        """Evaluate the logarithmic likelihood at the model parameter
        point.

        Parameters
        ----------
        param_point : float array_like
            A vector of all model parameters associated with
            :attr:`lumfunc_model` except the luminosity function arguments
            (luminosity/magnitude and redshift).
        use_prior : bool, optional
            If `True` (default is `False`), use the user-input prior
            (i.e. set ``-numpy.inf`` at parameter points outside the prior
            range).

        Returns
        -------
        float
            Logarithmic likelihood value.

        """
        if isinstance(param_point, np.ndarray):
            param_point = param_point.tolist()
        else:
            param_point = list(param_point)

        if use_prior:
            log_prior = _uniform_log_pdf(
                np.reshape(param_point, -1), list(self.prior.values())
            )

            if not np.isfinite(log_prior):
                return log_prior
        else:
            log_prior = 0.

        model_params = OrderedDict(zip(list(self.prior.keys()), param_point))
        if self.fixed is not None:
            model_params.update(self.fixed)

        model_vector = [
            self._lumfunc_model(
                *data_point, base10_log=self._lg_conversion, **model_params
            )
            for data_point in self.data_points
        ]

        log_likelihood = _normal_log_pdf(
            self._data_vector, model_vector, self._data_covariance
        )

        return log_prior + log_likelihood

    def _setup_prior(self):

        with open(self._prior_source_path, 'r') as file:
            parameter_names = list(
                map(
                    lambda header: header.strip(" "),
                    file.readline().strip("#").strip("\n").split(",")
                )
            )
        prior_data = np.genfromtxt(self._prior_source_path, unpack=True)
        prior_ranges = list(map(tuple, prior_data))
        prior = OrderedDict(zip(parameter_names, prior_ranges))

        if self._fixed_source_path is not None:
            with open(self._fixed_source_path, 'r') as file:
                fixed_names = list(
                    map(
                        lambda header: header.strip(" "),
                        file.readline().strip("#").strip("\n").split(",")
                    )
                )
            fixed_values = np.genfromtxt(self._fixed_source_path, unpack=True)
            fixed = OrderedDict(zip(fixed_names, fixed_values))
        else:
            fixed = None

        return prior, fixed

    def _setup_data_points(self):

        data_points_flat = list(map(
            lambda tup: tuple(reversed(tup)),
            iterproduct(self.redshift_bins, self.brightness_bins)
        ))

        return list(iterfilter(data_points_flat, self._point_validity))

    def __str__(self):

        return (
            "LuminosityFunctionLikelihood"
            "(data_source='{}',lum_func_model='{}')"
            .format(self._data_source_path, self._lumfunc_model.__name__)
        )
