r"""
Luminosity function likelihood (:mod:`~horizonground.lumfunc_likelihood`)
===========================================================================

Fit tracer luminosity functions and produce model constraints.

Data processing
---------------

.. autosummary::

    DataSourceError
    DataSourceWarning
    LumFuncMeasurements


Likelihood evaluation
---------------------

.. autosummary::

    LumFuncLikelihood

|

"""
import warnings
from collections import OrderedDict
from inspect import signature
from itertools import compress as iterfilter
from itertools import product as iterproduct

import numpy as np

from .utils import process_header


class DataSourceError(IOError):
    """Raise an exception when the data source is not available or not in
    the correct format.

    """


class DataSourceWarning(UserWarning):
    """Raise an exception when the data source does not appear to be
    self-consistent.

    """


class LumFuncMeasurements:
    """Luminosity function measurements for a tracer sample.

    Parameters
    ----------
    measurement_file : *str or* :class:`pathlib.Path`
        File path to luminosity function measurements.
    uncertainty_file : *str or* :class:`pathlib.Path` *or None, optional*
        File path to luminosity function uncertainties.
    base10_log : bool, optional
        If `True` (default), measurement values are converted to base-10
        logarithms.  String ``'lg_'`` is detected in file headers to
        determine whether loaded measurements are already in base-10
        logarithms.

    Attributes
    ----------
    luminosity_bins : list of float
        Luminosity bin centres.
    redshift_bins : list of float
        Redshift bin centres.
    redshift_labels : list of str
        Redshift bin labels.

    Notes
    -----
    Data files are assumed to be an array where rows correspond to
    increasing luminosity bins and columns correspond to increasing
    redshift bins.  The first line should be a comment line for column
    headings and the first column should be the luminosity bins.  Redshift
    bin labels are read from column headings as suffices in the
    format e.g. "z_0.5_1.5".  If symmetric uncertainties are given for
    base-10 logarithmic luminosity function and conversion to linear values
    is needed, the larger asymmetric uncertainty on the linear luminosity
    function value is taken.

    """

    def __init__(self, measurement_file, uncertainty_file=None,
                 base10_log=True):

        self._lg_conversion = base10_log
        self._measurement_source_path = measurement_file
        self._uncertainty_source_path = uncertainty_file

        self._load_source_files()

        self._valid_data_points = ~np.isnan(self._measurements.flatten())
        if self._uncertainties is not None:
            self._valid_data_points &= ~np.isnan(self._uncertainties.flatten())
            if not self._lg_conversion:
                self._valid_data_points &= np.greater_equal(
                    np.abs(self._measurements.flatten()),
                    np.abs(self._uncertainties.flatten())
                )

    def __str__(self):

        return "LuminosityFunctionMeasurements({})".format(",".join(
            [
                "measurement_source={}".format(self._measurement_source_path),
                "uncertainty_source={}".format(self._uncertainty_source_path),
            ]
        ))

    def __getitem__(self, z_key):
        """Get luminosity function measurements and uncertainties (if
        available) for a specific redshift bin.

        Parameters
        ----------
        z_key: int, slice or str
            Slice or integer index or string representing redshift
            bins.  If a string, the accepted format is e.g. ``z=1.0``.

        Returns
        -------
        :class:`numpy.ndarray`, :class:`numpy.ndarray` or None
            Measurements and uncertainties for the redshift bin(s).

        """
        if isinstance(z_key, (int, slice)):
            z_idx = z_key
        else:
            try:
                z = float(str(z_key).replace(" ", "").lstrip("z="))
                z_idx = np.where(np.isclose(self.redshift_bins, z))[0][0]
            except (TypeError, ValueError):
                raise KeyError(
                    "Non-existent redshift bin for '{}'.".format(z_key)
                )

        if self._uncertainties is not None:
            return self._measurements[z_idx], self._uncertainties[z_idx]
        return self._measurements[z_idx], None

    def get_statistics(self):
        """Return the empirical luminosity function data measurements
        and uncertainties as mean and variance statistics.

        Returns
        -------
        data_mean : :class:`numpy.ndarray`
            Luminosity function measurements as empirical mean.
        data_variance : :class:`numpy.ndarray` or None
            Luminosity function uncertainties as empirical variance,
            if available.

        Notes
        -----
        Data vectors are ordered by increasing redshift bins and
        within the same redshift bin ordered by increasing luminosity.

        """
        data_mean = self._measurements.flatten()[self._valid_data_points]

        if self._uncertainties is not None:
            data_variance = np.square(
                self._uncertainties.flatten()[self._valid_data_points]
            )
        else:
            data_variance = None

        return data_mean, data_variance

    def _load_source_files(self):

        # Process measurements.
        with open(self._measurement_source_path, 'r') as mfile:
            mheadings = process_header(mfile.readline(), skipcols=1)

        self.redshift_labels, self.redshift_bins = \
            self._extract_redshift_bins(mheadings)

        measurement_source = np.genfromtxt(
            self._measurement_source_path, unpack=True
        )

        self.luminosity_bins = measurement_source[0]

        measurement_array = measurement_source[1:]

        if len(measurement_array) != len(mheadings):
            raise DataSourceError(
                "Measurements do not match the number of headings ."
            )

        if not self._lg_conversion:
            for z_idx, col_name in enumerate(mheadings):
                if 'lg_' in col_name:
                    measurement_array[z_idx] = 10 ** measurement_array[z_idx]
                    mheadings[z_idx] = col_name.replace("lg_", "")

        self._measurements = measurement_array

        # Process uncertainties.
        if not self._uncertainty_source_path:
            self._uncertainties = None
        else:
            with open(self._uncertainty_source_path, 'r') as ufile:
                uheadings = process_header(ufile.readline(), skipcols=1)

            uncertainty_source = np.genfromtxt(
                self._uncertainty_source_path, unpack=True
            )

            luminosity_bin_matching_msg = (
                "Luminosity bins in measurement and uncertainty files "
                "do not match."
            )
            if len(self.luminosity_bins) != len(uncertainty_source[0]):
                raise DataSourceError(luminosity_bin_matching_msg)
            if not np.allclose(self.luminosity_bins, uncertainty_source[0]):
                warnings.warn(luminosity_bin_matching_msg, DataSourceWarning)

            uncertainty_array = uncertainty_source[1:].copy()

            if np.shape(uncertainty_array) != np.shape(measurement_array):
                raise DataSourceError(
                    "Uncertainty file data do not match measurement file data."
                )

            if not self._lg_conversion:
                # Optimistic wing of assymetric uncertainties.
                error_array = uncertainty_source[1:].copy()
                for z_idx, col_name in enumerate(uheadings):
                    if 'lg_' in col_name:
                        error_array[z_idx] = measurement_array[z_idx] \
                            * (1 - 10 ** (- error_array[z_idx]))
                        # Larger asymetric uncertainty is taken.
                        uncertainty_array[z_idx] = measurement_array[z_idx] \
                            * (10 ** uncertainty_array[z_idx] - 1)
                        uheadings[z_idx] = col_name.replace("lg_", "")
                self._errors = error_array

        self._uncertainties = uncertainty_array

    @staticmethod
    def _extract_redshift_bins(headings):

        bin_labels = sorted(set(map(
            r"${}$".format,
            [
                column.split("z_")[-1].replace("_", "<z<")
                for column in headings if 'z_' in column
            ]
        )))

        bin_centres = np.mean(
            [
                tuple(map(float, column.split("z_")[-1].split("_")))
                for column in headings if 'z_' in column
            ],
            axis=-1
        )

        return bin_labels, bin_centres


def _uniform_log_pdf(param_vals, param_ranges):

    if len(param_ranges) != len(param_vals):
        raise ValueError(
            "Number of parameter ranges does not match "
            "the number of parameters."
        )

    param_ranges = np.sort(np.atleast_2d(param_ranges), axis=-1)

    if any(np.less(param_vals, param_ranges[:, 0])) \
            or any(np.greater(param_vals, param_ranges[:, 1])):
        return - np.inf
    return 0.


def _normal_log_pdf(deviation_vector, covariance_matrix):

    deviation_vector = np.asarray(deviation_vector)

    if not all(np.isfinite(deviation_vector)):
        return - np.inf

    return - 1/2 * np.dot(
        deviation_vector, np.linalg.solve(covariance_matrix, deviation_vector)
    )


class LumFuncLikelihood(LumFuncMeasurements):
    """Luminosity function likelihood.

    Notes
    -----
    The built-in likelihood distribution is a multivariate normal
    approximation near the maximum point of the Poisson distribution that
    describes the tracer number count in redshift and luminosity bins,
    with different prescriptions of the luminosity function uncertainties
    to the diagonal covariance matrix of the multivariate normal
    distribution (see appendix B of [1]_).  The built-in prior
    distribution is multivariate uniform.

    .. [1] Pozzetti L. et al., 2016. A&A 590, A3.
       [arXiv: `1603.01453 <https://arxiv.org/abs/1603.01453>`_]


    Parameters
    ----------
    model_lumfunc : callable
        Luminosity function model.  Must return base-10 logarithmic values
        or accept `base10_log` boolean keyword argument.
    measurement_file : str or :class:`pathlib.Path`
        Luminosity function measurement file path.
    prior_file : str or :class:`pathlib.Path`
        Luminosity function model prior file path.  The prior parameter
        values (parameter ranges) may not matter, but the parameter names
        provided in the file do.
    model_constraint : callable or None, optional
        Additional model constraint(s) to be imposed on model parameters
        as a prior (default is `None`).
    uncertainty_file : str or :class:`pathlib.Path` *or None, optional*
        Luminosity function uncertainty file path (default is `None`).
        Ignored if `data_covariance` is provided.
    fixed_file : str or :class:`pathlib.Path` *or None, optional*
        Luminosity function model fixed parameter file path (default is
        `None`).  This covers any model parameter(s) not included in
        the prior.
    prescription : {'native', 'poisson', 'symlg', 'symlin'}, str, optional
        Gaussian likelihood approximation prescription (default is
        'poisson').
    model_options : dict or None, optional
        Additional parameters passed to the `model_lumfunc` for model
        evaluation (default is `None`).  This should not contain
        parametric luminosity function model parameters but only Python
        function implementation optional parameters (e.g.
        ``redshift_pivot=2.2`` for
        :func:`~horizonground.lumfunc_modeller.quasar_PLE_lumfunc`).
    covariance_matrix : float array_like or None, optional
        Covariance matrix for the multivariate normal likelihood
        approximation (default is `None`).  Its dimensions must match the
        length of the data vector flattened by redshift and
        luminosity bins.

    Attributes
    ----------
    data_points : list of float
        A vector of (luminosity, redshift) points for each valid
        luminosity function measurement.
    data_vector : float :class:`numpy.ndarray`
        Flattened luminosity function measurements for valid data points.
    prior, fixed : :class:`collections.OrderedDict` or None
        Ordered dictionary of varied prior parameter names and values or
        fixed parameter names and values.  The parameters are ordered
        as in the input files, so that the ordering of luminosity function
        arguments is consistent.

    """

    def __init__(self, model_lumfunc, measurement_file, prior_file,
                 uncertainty_file=None, fixed_file=None,
                 prescription='poisson', model_constraint=None,
                 model_options=None, covariance_matrix=None):

        # Set up likelihood treatment.
        self._prior_source_path = prior_file
        self._fixed_source_path = fixed_file
        self.prior, self.fixed = self._setup_prior()

        self._presciption = prescription
        if self._presciption in ['native']:
            self._lg_conversion = False
        elif self._presciption in ['poisson', 'symlg', 'symlin']:
            self._lg_conversion = True
        else:
            raise ValueError(
                f"Invalid Gaussian likelihood prescription: {prescription}."
            )

        # Set up model evaluation.
        self._model_lumfunc = model_lumfunc
        self._model_constraint = model_constraint
        self._model_options = model_options or {}

        if 'base10_log' in signature(self._model_lumfunc).parameters:
            self._model_options.update({'base10_log': self._lg_conversion})

        # Set up data processing.
        super().__init__(
            measurement_file, uncertainty_file=uncertainty_file,
            base10_log=self._lg_conversion
        )
        self.data_points = self._setup_data_points()
        self.data_vector, self._covariance = \
            self._get_moments(external_covariance=covariance_matrix)

    def __str__(self):

        return "LuminosityFunctionLikelihood({})".format(",".join(
            [
                "LF_model={}".format(self._model_lumfunc.__name__),
                "measurement_source={}".format(self._measurement_source_path),
                "uncertainty_source={}".format(self._uncertainty_source_path),
                "prior_source={}".format(self._prior_source_path),
                "fixed_source={}".format(self._fixed_source_path),
                "likelihood_approximant={}".format(self._presciption),
            ]
        ))

    def __call__(self, param_point, use_prior=False):
        """Evaluate the logarithmic likelihood at the model parameter
        point.

        Parameters
        ----------
        param_point : float array_like
            :attr:`model_lumfunc` parametric model parameters ordered in
            the same way as the prior parameters.
        use_prior : bool, optional
            If `True` (default is `False`), use the user-input prior range.

        Returns
        -------
        float
            Logarithmic likelihood value.

        """
        if isinstance(param_point, np.ndarray):
            param_point = param_point.tolist()
        else:
            param_point = list(param_point)

        # Check for prior.
        if use_prior:
            log_prior = _uniform_log_pdf(
                np.reshape(param_point, -1), list(self.prior.values())
            )
        else:
            log_prior = 0.

        if not np.isfinite(log_prior):
            return - np.inf

        # Check for model constraint.
        model_params = OrderedDict(zip(list(self.prior.keys()), param_point))
        if self.fixed is not None:
            model_params.update(self.fixed)

        try:
            within_constraint = self._model_constraint(model_params)
        except TypeError:
            within_constraint = True

        if not within_constraint:
            return - np.inf

        # Pass full parameter scope to the luminosity function model.
        model_params.update(self._model_options)

        model_vector = [
            self._model_lumfunc(*data_point, **model_params)
            for data_point in self.data_points
        ]

        # Form Gaussian approximant likelihood deviation vector.
        deviation_vector = np.subtract(model_vector, self.data_vector)
        covariance_matrix = self._covariance

        if self._presciption == 'poisson':
            deviation_vector = np.sqrt(
                2 * np.abs(np.exp(deviation_vector) - 1 - deviation_vector)
            )
        elif self._presciption == 'symlin':
            deviation_vector = np.exp(deviation_vector) - 1

        log_likelihood = _normal_log_pdf(deviation_vector, covariance_matrix)

        return log_prior + log_likelihood

    def _setup_data_points(self):

        # The primary axis is redshift, secondary luminosity.
        _data_points = list(map(
            lambda tup: tuple(reversed(tup)),
            iterproduct(self.redshift_bins, self.luminosity_bins)
        ))

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

    def _get_moments(self, external_covariance=None):

        data_mean, data_var = self.get_statistics()

        if external_covariance is not None:
            data_covar = np.squeeze(external_covariance)
            if len(set(np.shape(data_covar))) > 1 \
                    or len(data_covar) != len(self.data_points):
                raise ValueError(
                    "`covariance_matrix` dimensions do not match data points: "
                    "({:d}, {:d}) versus {:d}."
                    .format(*np.shape(data_covar), len(self.data_points))
                )
            return data_mean, data_covar

        if data_var is None:
            raise ValueError(
                "Either `uncertainty_file` or `covariance_matrix` must be "
                "provided for setting the likelihood covariance matrix."
            )

        return data_mean, np.diag(data_var)
