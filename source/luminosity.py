r"""
Luminosity Modeller (:mod:`luminosity`)
===========================================================================

Model luminosity function and related quantities.

Pure luminosity evolution (PLE)
-------------------------------

The *quasar* luminosity function in the pure luminosity evolution (PLE)
model is a double power law

.. math::

    \Phi(m, z) = \frac{\Phi^\ast}{
        10^{0.4 (\alpha + 1) [m - m^\ast(z)]}
        + 10^{0.4 (\beta + 1) [m - m^\ast(z)]}
    }

where :math:`m` is the absolute magnitude suitably normalised, :math:`z`
is the redshift, and the slope parameters :math:`\alpha, \beta` describe
the power law on the bright and faint ends respectively, but their values
differ below and above the pivot redshift :math:`z > z_\textrm{p}`.
:math:`m^\ast` is the break magnitude at which the luminosity function
evaluates to :math:`\Phi^\ast`,

.. math::

    m^\ast(z) = m^\ast(z_\textrm{p}) - \frac{5}{2} [
        k_1 (z - z_\textrm{p}) + k_2 (z - z_\textrm{p})^2
    ] \,,

where :math:`k_1, k_2` are the redshift-evolution parameters whose values
also differ below and above the pivot redshift.

This is a parametric model with 10 parameters: :math:`\Phi^\ast`,
:math:`m^\ast(z_\textrm{p}`, :math:`(\alpha, \beta, k_1, k_2)_\textrm{l}`
for :math:`z < z_\textrm{p}` and
:math:`(\alpha, \beta, k_1, k_2)_\textrm{h}` for :math:`z > z_\textrm{p}`.


Hybrid evolution (PLE+LEDE)
---------------------------

.. todo:: Not implemented.

"""
import numpy as np
from scipy.integrate import quad
from scipy.misc import derivative


def quasar_luminosity_PLE_model(magnitude, redshift, redshift_pivot=2.2,
                                **model_parameters):
    """Evaluate the pure luminosity evolution (PLE) model for the quasar
    luminosity function at the given absolute magnitude and redshift.

    Notes
    -----
    Magnitude is absolute and measured in :math:`g`-band normalised to
    the value at redshift 2.

    Parameters
    ----------
    magnitude : float
        Quasar magnitude.
    redshift : float
        Quasar redshift.
    redshift_pivot : float, optional
        Pivot redshift.
    **model_parameters
        PLE model parameters.

    Returns
    -------
    comoving_density : float :class:`numpy.ndarray`
        Predicted qausar comoving number density per unit magnitude (in
        inverse cubic Mpc).

    """
    # Re-definitions.
    M_g, z, z_p = magnitude, redshift, redshift_pivot

    # Determine the redshift range.
    if z <= z_p:
        subscript = '\\textrm{{{}}}'.format('l')
    else:
        subscript = '\\textrm{{{}}}'.format('h')

    # Set parameters.
    Phi_star = 10**model_parameters['\\log\\Phi^\\ast']
    M_g_star_p = model_parameters['M^\\ast_g(z_\\textrm{pivot})']

    alpha = model_parameters['\\alpha_{}'.format(subscript)]
    beta = model_parameters['\\beta_{}'.format(subscript)]

    k_1 = model_parameters['k_{{1{}}}'.format(subscript)]
    k_2 = model_parameters['k_{{2{}}}'.format(subscript)]

    # Evaluate the model prediction.
    exponent_magnitude_factor = M_g  - (
        M_g_star_p - 2.5*(k_1 * (z - z_p) + k_2 * (z - z_p)**2)
    )

    faint_power_law = 10 ** (0.4*(alpha + 1) * exponent_magnitude_factor)
    bright_power_law = 10 ** (0.4*(beta + 1) * exponent_magnitude_factor)

    comoving_density = Phi_star / (faint_power_law + bright_power_law)

    return comoving_density


def quasar_luminosity_hybrid_model(magnitude, redshift, redshift_pivot=2.2,
                                   **model_parameters):
    """Evaluate the hybrid model (pure luminosity evolution and luminosity
    evolution--density evolution, 'PLE+LEDE') for the quasar luminosity
    function at the given magnitude and redshift.

    Notes
    -----
    Magnitude is absolute and measured in :math:`g`-band normalised to
    the value at redshift 2.

    Parameters
    ----------
    magnitude : float
        Quasar magnitude.
    redshift : float
        Quasar redshift.
    redshift_pivot : float, optional
        Pivot redshift.
    **model_parameters
        'PLE+LEDE' hybrid model parameters.

    Returns
    -------
    comoving_density : float :class:`numpy.ndarray`
        Predicted qausar comoving number density per unit magnitude (in
        inverse cubic Mpc).

    """
    raise NotImplementedError


class LuminosityFunctionModeller:
    """Luminosity function modeller predicting the comoving number density
    and related quantities.

    Parameters
    ----------
    luminosity_model : callable
        A parametric model for the luminosity function with magnitude,
        redshift and additional arguments (in that order).
    threshold_magnitude : float
        Magnitude threshold.
    **model_parameters
        Keyword arguments to be passed to `luminosity_model`.

    Attributes
    ----------
    luminosity_function : callable
        Luminosity function of magnitude and redshift arguments only (in
        that order) (in inverse cubic Mpc).
    threshold_magnitude : float
        Magnitude threshold.
    model_parameters : dict
        Model parameters.

    """

    # HINT: Instead of ``-np.inf`` to prevent arithmetic overflow.
    MAGNITUDE_LIMIT = -40.
    """float: Finite magnitude upper limit .

    """

    def __init__(self, luminosity_model, threshold_magnitude,
                 **model_parameters):

        self.luminosity_function = lambda m, z: \
            luminosity_model(m, z, **model_parameters)
        self.threshold_magnitude = threshold_magnitude
        self.model_parameters = model_parameters

        self._comoving_number_density = None
        self._evolution_bias = None
        self._magnification_bias = None

    @classmethod
    def from_parameters_file(cls, luminosity_model, threshold_magnitude,
                             file_path):
        """Instantiate the modeller by loading model parameter values from
        a file.

        Parameters
        ----------
        luminosity_model : callable
            A parametric model for the luminosity function with magnitude,
            redshift and additional arguments (in that order).
        threshold_magnitude : float
            Magnitude threshold.
        file_path : str
            Path of the model parameter file.

        """
        with open(file_path, 'r') as pfile:
            parameters = tuple(
                map(
                    lambda var_name: var_name.strip(" "),
                    pfile.readline().strip("#").strip("\n").split(",")
                )
            )
            estimates = tuple(
                map(
                    lambda value: float(value),
                    pfile.readline().split(",")
                )
            )

        return cls(
            luminosity_model, threshold_magnitude,
            **dict(zip(parameters, estimates))
        )

    @property
    def comoving_number_density(self):
        """Comoving number density (in inverse cubic Mpc) as a function of
        redshift.

        Returns
        -------
        callable

        """
        if callable(self._comoving_number_density):
            return self._comoving_number_density

        self._comoving_number_density = lambda z: quad(
            self.luminosity_function,
            self.MAGNITUDE_LIMIT,
            self.threshold_magnitude,
            args=(z,)
        )[0]

        return self._comoving_number_density

    @property
    def evolution_bias(self):
        """Evolution bias as a function of redshift.

        Returns
        -------
        callable

        """
        if callable(self._evolution_bias):
            return self._evolution_bias

        ln_comoving_number_density = lambda z: np.log(
            self.comoving_number_density(z)
        )

        self._evolution_bias = lambda z: 3 - (1 + z) * derivative(
            ln_comoving_number_density, z, dx=1e-2
        )

        return self._evolution_bias

    @property
    def magnification_bias(self):
        """Magnification bias as a function of redshift.

        Returns
        -------
        callable

        """
        if callable(self._magnification_bias):
            return self._magnification_bias

        self._magnification_bias = lambda z: \
            self.luminosity_function(self.threshold_magnitude, z) \
            / (np.log(10) * self.comoving_number_density(z))

        return self._magnification_bias
