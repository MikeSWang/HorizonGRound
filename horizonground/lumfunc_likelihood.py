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

import warnings
from collections import OrderedDict
from itertools import compress as iterfilter
from itertools import product as iterproduct

import numpy as np

from .utils import process_header


class DataSourceError(Exception):
    """Data source error.

    """
    pass


class DataSourceWarning(UserWarning):
    """Data source warning.

    """
    pass


class LumFuncMeasurements:
    """Luminosity function measurements for a tracer sample.

    Parameters
    ----------
    measurements_file : str or :class:`pathlib.Path`
        Luminosity function measurements file path.
    uncertainties_file : str or :class:`pathlib.Path`
        Luminosity function uncertainties file path.
    base10_log : bool, optional
        If `True` (default), all values are converted to base-10
        logarithms.

    Attributes
    ----------
    brightness_bins : list of float
        Luminosity/magnitude bin centres.
    redshift_bins : list of float
        Redshift bin centres.
    redshift_labels : list of str
        Redshift bin labels.

    Notes
    -----
    Data files are assumed to be an array where rows correspond to
    increasing luminosity/magnitude bins (except the first line for
    headings) and columns correspond to increasing redshift bins
    (except the first column for luminosity/magnitude bin centres).
    Redshift bins are attached to column headings as suffices in the
    format e.g. "z_0.5_1.5".

    """

    def __init__(self, measurements_file, uncertainties_file=None,
                 base10_log=True):

        self._lg_conversion = base10_log
        self._measurements_source_path = measurements_file
        self._uncertainties_source_path = uncertainties_file
        self._load_source_files()

        self._valid_data_points = ~np.isnan(self._measurements.flatten())
        if self._uncertainties is not None:
            self._valid_data_points &= ~np.isnan(self._uncertainties.flatten())

    def get_statistics(self):
        """Return the empirical mean, and covariance if available, from
        luminosity function data.

        Returns
        -------
        data_mean : :class:`numpy.ndarray`
            Empirical mean.
        data_variance : :class:`numpy.ndarray` or None
            Empirical variance, if available.

        Notes
        -----
        Data vectors are ordered by increasing redshift bins and
        within the same redshift bin ordered by increasing
        luminosity/magnitude.

        """
        data_mean = self._measurements.flatten()[self._valid_data_points]

        if self._uncertainties is not None:
            data_variance = \
                self._uncertainties.flatten()[self._valid_data_points]
        else:
            data_variance = None

        return data_mean, data_variance

    def __getitem__(self, z_key):
        """Get luminosity function measurements, and uncertainties if
        available, for a specific redshift bin.

        Parameters
        ----------
        z_key: int, slice or str
            Slice or integer index or string representing a redshift
            bin.  If a string, the accepted format is e.g. ``z=1.0``.

        Returns
        -------
        :class:`numpy.ndarray`, :class:`numpy.ndarray` or None
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

        if self._uncertainties is not None:
            return self._measurements[z_idx], self._uncertainties[z_idx]
        return self._measurements[z_idx], None

    def _load_source_files(self):

        # Process measurements.
        source_measurements = np.genfromtxt(
            self._measurements_source_path, unpack=True
        )

        self.brightness_bins = source_measurements[0]

        measurements_array = source_measurements[1:]

        with open(self._measurements_source_path, 'r') as mfile:
            mheadings = process_header(mfile.readline())

        if len(mheadings) != len(measurements_array):
            raise DataSourceError(
                "Number of headings does not match measurements. "
            )

        self.redshift_labels, self.redshift_bins = \
            self._extract_redshift_bins(mheadings)

        if not self._lg_conversion:
            for z_idx, col_name in enumerate(mheadings):
                if 'lg_' in col_name:
                    measurements_array[z_idx] = 10 ** measurements_array[z_idx]
                    mheadings[z_idx] = col_name.replace("lg_", "")

        # Process uncertainties.
        if not self._uncertainties_source_path:
            self._uncertainties = None
        else:
            source_uncertainties = np.genfromtxt(
                self._uncertainties_source_path, unpack=True
            )

            brightness_bin_matching_msg = (
                "Brightness bins in measurements and uncertainties files "
                "do not match. "
            )
            if len(self.brightness_bins) != len(source_uncertainties[0]):
                raise DataSourceError(brightness_bin_matching_msg)
            if not np.allclose(self.brightness_bins, source_uncertainties[0]):
                warnings.warn(brightness_bin_matching_msg, DataSourceWarning)

            uncertainties_array = source_uncertainties[1:]

            if np.shape(measurements_array) != np.shape(uncertainties_array):
                raise DataSourceError(
                    "Uncertainties file data do not match "
                    "measurements file data."
                )

            with open(self._uncertainties_source_path, 'r') as ufile:
                uheadings = process_header(ufile.readline())

            if not self._lg_conversion:
                for z_idx, col_name in enumerate(uheadings):
                    if 'lg_' in col_name:
                        uncertainties_array[z_idx] = \
                            measurements_array[z_idx] \
                            * (10 ** uncertainties_array[z_idx] - 1)
                        uheadings[z_idx] = col_name.replace("lg_", "")

        self._measurements = measurements_array
        self._uncertainties = uncertainties_array

    def _extract_redshift_bins(self, data_headings):

        bin_labels = self._sort_unique_elements(
            map(
                r"${}$".format,
                [
                    dname.split("z_")[-1].replace("_", "<z<")
                    for dname in data_headings if 'z_' in dname
                ]
            )
        )

        bin_centres = [
            np.mean(tuple(map(float, blabel.strip("$").split("<z<"))))
            for blabel in bin_labels
        ]

        return bin_labels, bin_centres

    @staticmethod
    def _sort_unique_elements(sequence):
        return sorted(list(set(sequence)))

    def __str__(self):

        return (
            "LuminosityFunctionMeasurements(data_source='{}')"
            .format(self._measurements_source_path)
        )


def _uniform_log_pdf(param_vals, param_ranges):

    if len(param_ranges) != len(param_vals):
        raise ValueError(
            "Number of parameter ranges does not match number of parameters. "
        )
    if any(np.greater(param_ranges[:, 0], param_ranges[:, 1])):
        param_ranges = np.sort(param_ranges, axis=-1)
        warnings.warn("Uniform prior ranges reordered. ")

    if any(np.less(param_vals, param_ranges[:, 0])) \
            or any(np.greater(param_vals, param_ranges[:, 1])):
        return - np.inf
    return 0.


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
    measurements_file : str or :class:`pathlib.Path`
        Luminosity function measurements file path.
    prior_file : str or :class:`pathlib.Path`
        Luminosity function model prior file path.  The prior parameter
        values (parameter ranges) may not matter, but the parameter names
        provided in the file do.
    uncertainties_file : str or :class:`pathlib.Path` or None, optional
        Luminosity function uncertainties file path (default is `None`).
        Ignored if `data_covariance` is provided.
    fixed_file : str or :class:`pathlib.Path` or None, optional
        Luminosity function model fixed parameter file path.  This covers
        any model parameter(s) not included in the prior.
    data_covariance : float array_like or None, optional
        Covariance matrix for the data points.  Its dimensions must match
        the length of the data vector for valid data points ordered by
        brightness and redshift.
    model_constraint : callable or None, optional
        Additional model constraint(s) to be imposed on model parameters
        as a prior (default is `None`).
    base10_log : bool, optional
        If `True` (default), all values are converted to base-10
        logarithms.

    Attributes
    ----------
    data_points : list of float
        A vector of (brightness, redshift) coordinates for each valid
        luminosity function measurement.
    prior, fixed : :class:`collections.OrderedDict` or None
        Ordered dictionary of varied prior parameter names and values or
        fixed parameter names and values.  The parameters are ordered
        as in the input files, so the ordering of likelihood function
        arguments is consistently the same.

    """

    def __init__(self, lumfunc_model, measurements_file, prior_file,
                 uncertainties_file=None, fixed_file=None,
                 data_covariance=None, model_constraint=None, base10_log=True):

        super().__init__(
            measurements_file,
            uncertainties_file=uncertainties_file, base10_log=base10_log
        )

        self._lumfunc_model = lumfunc_model
        self._model_constraint = model_constraint

        self.data_points = self._setup_data_points()

        self._prior_source_path = prior_file
        self._fixed_source_path = fixed_file
        self.prior, self.fixed = self._setup_prior()

        self._external_data_covariance = data_covariance
        self._data_vector, self._data_covariance = self._get_moments()

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

        if callable(self._model_constraint):
            if not self._model_constraint(model_params):
                return - np.inf

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

    def _setup_data_points(self):

        # The primary axis is brightness, secondary redshift.
        _data_points = list(
            map(
                lambda tup: tuple(reversed(tup)),
                iterproduct(self.redshift_bins, self.brightness_bins)
            )
        )

        return list(iterfilter(_data_points, self._valid_data_points))

    def _setup_prior(self):

        with open(self._prior_source_path, 'r') as pfile:
            parameter_names = process_header(pfile.readline())

        prior_data = np.genfromtxt(self._prior_source_path, unpack=True)
        prior_ranges = list(map(tuple, prior_data))
        prior = OrderedDict(zip(parameter_names, prior_ranges))

        if self._fixed_source_path is None:
            fixed = None
        else:
            with open(self._fixed_source_path, 'r') as ffile:
                fixed_names = process_header(ffile.readline())
            fixed_values = np.genfromtxt(self._fixed_source_path, unpack=True)
            fixed = OrderedDict(zip(fixed_names, fixed_values))

        return prior, fixed

    def _get_moments(self):

        _data_mean, _data_var = self.get_statistics()

        if self._external_data_covariance:
            _data_covar = np.squeeze(self._external_data_covariance)
            if len(set(np.shape(_data_covar))) > 1 \
                    or len(_data_covar) != len(self.data_points):
                raise ValueError(
                    "`data_covariance` dimensions do not match data points: "
                    "({:d}, {:d}) versus {:d}."
                    .format(len(_data_covar), len(self.data_points))
                )
            return _data_mean, _data_covar

        if _data_var is None:
            raise ValueError(
                "Either `uncertainties_file` or `data_covariance` must be "
                "provided for setting the covariance matrix "
                "in the likelihood distribution."
            )

        return _data_mean, np.diag(_data_var)

    def __str__(self):

        return (
            "LuminosityFunctionLikelihood"
            "(measurements_source='{}',LF_model='{}')"
            .format(self._measurements_source_path, self._lumfunc_model.__name__)
        )
