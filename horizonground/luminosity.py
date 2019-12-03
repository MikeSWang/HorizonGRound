r"""
Luminosity Modeller (:mod:`luminosity`)
===========================================================================

Model tracer luminosity functions and related quantities.


Pure luminosity evolution (PLE)
-------------------------------

The *quasar* luminosity function in the pure luminosity evolution (PLE)
model is a double power law

.. math::

    \Phi(m, z) = \frac{\Phi_\ast}{
        10^{0.4 (\alpha + 1) [m - m_\ast(z)]}
        + 10^{0.4 (\beta + 1) [m - m_\ast(z)]}
    }

where :math:`m` is the absolute magnitude suitably normalised, :math:`z`
is the redshift, and the slope parameters :math:`\alpha, \beta` describe
the power law on the bright and faint ends respectively, but their values
differ below and above the pivot redshift :math:`z > z_\textrm{p}`.
:math:`m_\ast` is the break magnitude at which the luminosity function
evaluates to :math:`\Phi_\ast`,

.. math::

    m_\ast(z) = m_\ast(z_\textrm{p}) - \frac{5}{2} \left[
        k_1 (z - z_\textrm{p}) + k_2 (z - z_\textrm{p})^2
    \right] \,,

where :math:`k_1, k_2` are the redshift-evolution parameters whose values
also differ below and above the pivot redshift.

This is a parametric model with 10 parameters: :math:`\Phi_\ast`,
:math:`m_\ast(z_\textrm{p})`, :math:`(\alpha, \beta, k_1, k_2)_\textrm{l}`
for :math:`z < z_\textrm{p}` and
:math:`(\alpha, \beta, k_1, k_2)_\textrm{h}` for :math:`z > z_\textrm{p}`.


Hybrid evolution (PLE+LEDE)
---------------------------

.. todo:: Not implemented.


Schechter function
------------------

The H |alpha| -emitter luminosity function in the Schechter model takes the
form of a gamma function

.. math::

    \Phi(L, z) \operatorname{d}\!L = \underbrace{
        \ln10 \, \Phi_\ast(z) y(z)^{\alpha+1} \mathrm{e}^{-y(z)}
    }_{\Phi(\lg{L}, z)} \operatorname{d}\!\lg{L}

where :math:`\alpha` is the faint-end slope parameter,

.. math::
    \begin{align*}
        \Phi_\ast(z) &=
            \begin{cases}
                \Phi_{\ast0} (1 + z)^\epsilon \,,
                    \quad z \leqslant z_\textrm{b} \,; \\
                \Phi_{\ast0} (1 + z_\textrm{b})^{2\epsilon}
                    (1 + z)^{-\epsilon} \,, \quad z > z_\textrm{b} \,,
            \end{cases} \\
        y(z) &= \frac{L}{L_{\ast0}} (1 + z)^{-\delta} \,,
    \end{align*}

are the redshift-dependent characteristic comoving number density and
relative luminosity of the H |alpha| -emitters, :math:`\epsilon, \delta`
are the redshift-evolution indices, and :math:`z_\textrm{b}` is the break
magnitude.

This is a parametric model with 6 parameters: :math:`\alpha, \epsilon,
\delta`, :math:`z_\textrm{b}`, :math:`m_{\ast0}` and :math:`\Phi_{\ast0}`.

.. |alpha| unicode:: U+03B1
    :trim:

"""
import numpy as np
from astropy import units
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
    Phi_star = 10**model_parameters['\\lg\\Phi_\\ast']
    M_g_star_p = model_parameters['M_{g\\ast}(z_\\textrm{p})']

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


def alpha_emitter_luminosity_schechter_model(lg_luminosity, redshift,
                                             **model_parameters):
    r"""Evaluate the Schechter model for the H |alpha| -emitter
    luminosity function at the given base-10 logarithmic luminosity and
    redshift.

    Parameters
    ----------
    lg_luminosity : float
        Emitter luminosity in base-10 logarithm.
    redshift : float
        Emitter redshift.
    **model_parameters
        Schechter model parameters.

    Returns
    -------
    comoving_density : float :class:`numpy.ndarray`
        Predicted emitter comoving number density per unit base-10
        logarithmic luminosity (in inverse cubic Mpc).

    """
    # Re-definitions.
    y0 = 10**(lg_luminosity - model_parameters['\\lg{L_{\\ast0}}'])
    z = redshift

    # Set parameters.
    Phi_star0 = 10**model_parameters['\\lg\\Phi_{\\ast0}']
    z_b = model_parameters['z_\\textrm{b}']

    alpha = model_parameters['\\alpha']
    delta = model_parameters['\\delta']
    epsilon = model_parameters['\\epsilon']

    # Evaluate the model prediction.
    if z <= z_b:
        Phi_star = Phi_star0 * (1 + z)**epsilon
    else:
        Phi_star = Phi_star0 * (1 + z_b)**(2*epsilon) / (1 + z)**epsilon

    y = y0 / (1 + z)**delta

    comoving_density = np.log(10) * Phi_star * y**(alpha + 1) * np.exp(-y)

    return comoving_density


class LuminosityFunctionModeller:
    r"""Luminosity function modeller predicting the comoving number density
    and related quantities for a given brightness threshold.

    Notes
    -----
    All luminosity values are represented in base-10 logarithms.

    Parameters
    ----------
    luminosity_model : callable
        A parametric model for the luminosity function of luminosity or
        magnitude and redshift variables accepting additional parameters
        (in that order).
    luminosity_variable : {'luminosity', 'magnitude'}, str
        Luminosity variable, either ``'luminosity'`` or ``'magnitude'``.
    threshold_value : float
        Luminosity threshold value.
    threshold_variable : {'flux', 'luminosity', 'magnitude'}, str
        Luminosity threshold variable, one of ``'flux'``, ``'luminosity'``
        or ``'magnitude'``.  If ``'flux'``, `threshold_value` will be
        converted into ``'luminosity'`` threshold value.
    cosmology : :class:`astropy.cosmology.Cosmology` or None, optional
        Background cosmological model (default is `None`).  If
        `threshold_variable` is 'flux', this is needed for computing the
        luminosity distance and cannot be `None`.
    **model_parameters
        Additional model parameters to be passed to `luminosity_model` as
        keyword arguments.

    Attributes
    ----------
    model_parameters : dict
        Model parameters.
    luminosity_function : callable
        Luminosity function of luminosity or magnitude and redshift
        variables only (in that order) (in inverse cubic Mpc).
    luminosity_variable : {'luminosity', 'magnitude'}, str
        Luminosity variable, either ``'luminosity'`` or ``'magnitude'``.
    luminosity_threshold : callable
        Luminosity threshold in :attr:`luminosity_variable` as a function
        of redshift.

    Raises
    ------
    ValueError
        If `cosmology` is `None` when `threshold_variable` is ``'flux'``.

    """

    # HINT: Instead of ``np.inf`` to prevent arithmetic overflow.
    brightness_bound = {
        'luminosity': 60.,
        'magnitude': -40.,
    }
    r"""float: Finite brightness bound.

    If :attr:`threshold_variable` is ``'luminosity'``, it is the base-10
    logarithmic value in erg/s; else if it is ``'magnitude'`` and
    dimensionless.

    """

    redshift_stepsize = 0.01
    r"""float: Redshift step size for numerical computations.

    """

    def __init__(self, luminosity_model, luminosity_variable, threshold_value,
                 threshold_variable, cosmology=None, **model_parameters):

        self._threshold_variable = self._alias(threshold_variable)
        if self._threshold_variable == 'flux':
            if cosmology is None:
                raise ValueError (
                    "`cosmology` cannot be None "
                    "if `threshold_variable` is 'flux'. "
                )
            self.luminosity_threshold = lambda z: np.log10(
                4*np.pi
                * threshold_value
                * cosmology.luminosity_distance(z).to(units.cm).value**2
            )
            self._threshold_variable = 'luminosity'
        else:
            self.luminosity_threshold = lambda z: threshold_value

        self.luminosity_variable = self._alias(luminosity_variable)
        if self._threshold_variable != self.luminosity_variable:
            raise NotImplementedError(
                "`threshold_variable` '{}' does not match "
                "`luminosity_variable` '{}'. "
                .format(self._threshold_variable, self.luminosity_variable)
            )

        self.model_parameters = model_parameters
        self.luminosity_function = np.vectorize(
            lambda lum, z: luminosity_model(lum, z, **self.model_parameters)
        )

        self._comoving_number_density = None
        self._evolution_bias = None
        self._magnification_bias = None

    @classmethod
    def from_parameters_file(cls, parameter_file, luminosity_model,
                             luminosity_variable, threshold_value,
                             threshold_variable, cosmology=None):
        """Instantiate the modeller by loading model parameter values from
        a file.

        Parameters
        ----------
        parameter_file : str
            Path of the model parameter file.
        luminosity_model : callable
            A parametric model for the luminosity function of luminosity or
            magnitude and redshift variables accepting additional
            parameters (in that order).
        luminosity_variable : {'luminosity', 'magnitude'}, str
            Luminosity variable, either ``'luminosity'`` or
            ``'magnitude'``.
        threshold_value : float
            Luminosity threshold value.
        threshold_variable : {'flux', 'luminosity', 'magnitude'}, str
            Luminosity threshold variable, one of ``'flux'``,
            ``'luminosity'`` or ``'magnitude'``.  If ``'flux'``,
            `threshold_value` will be converted into ``'luminosity'``
            threshold value.
        cosmology : :class:`astropy.cosmology.Cosmology` or None, optional
            Background cosmological model (default is `None`).  If
            `threshold_variable` is 'flux', this is needed for computing
            the luminosity distance and cannot be `None`.

        """
        with open(parameter_file, 'r') as pfile:
            parameters = tuple(
                map(
                    lambda var_name: var_name.strip(" "),
                    pfile.readline().strip("#").strip("\n").split(",")
                )
            )
            estimates = tuple(
                map(lambda value: float(value), pfile.readline().split(","))
            )

        return cls(
            luminosity_model, luminosity_variable,
            threshold_value, threshold_variable, cosmology=cosmology,
            **dict(zip(parameters, estimates))
        )

    @property
    def comoving_number_density(self):
        r"""Comoving number density (in inverse cubic Mpc) as a function of
        redshift

        .. math::

            \bar{n}(z;\bar{m}) = \int_{-\infty}^{\bar{m}}
                \operatorname{d}\!m \Phi(m, z)
            \quad \textrm{or} \quad
            \bar{n}(z;\lg\bar{L}) = \int^{\infty}_{\lg\bar{L}}
                \operatorname{d}\!\lg{L} \Phi(\lg{L}, z) \,.

        Returns
        -------
        callable
            Comoving number density (in inverse cubic Mpc).

        """
        if callable(self._comoving_number_density):
            return self._comoving_number_density

        self._comoving_number_density = lambda z: np.abs(
            quad(
                self.luminosity_function,
                self.luminosity_threshold(z),  # faint end
                self.brightness_bound[self.luminosity_variable],  # bright end
                args=(z,)
            )[0]
        )

        return np.vectorize(self._comoving_number_density)

    @property
    def evolution_bias(self):
        r"""Evolution bias as a function of redshift

        .. math::

            f_\textrm{ev}(z) = - (1 + z)
                \frac{\partial \ln\bar{n}(z)}{\partial z} \,.

        Returns
        -------
        callable
            Evolution bias.

        """
        if callable(self._evolution_bias):
            return self._evolution_bias

        ln_comoving_number_density = lambda z: np.log(
            self.comoving_number_density(z)
        )

        self._evolution_bias = lambda z: - (1 + z) * derivative(
            ln_comoving_number_density, z, dx=self.redshift_stepsize
        )

        return np.vectorize(self._evolution_bias)

    @property
    def magnification_bias(self):
        r"""Magnification bias as a function of redshift

        .. math::

            s(z) = \frac{1}{\ln10} \frac{
                \Phi(\bar{m},z)
            }{
                \bar{n}(z;\bar{m})
            }
            \quad \textrm{or} \quad
            s(z) = \frac{2}{5\ln10} \frac{
                \Phi(\lg\bar{L},z)
            }{
                \bar{n}(z;\bar{L})
            } \,.

        Returns
        -------
        callable
            Magnification bias.

        """
        if callable(self._magnification_bias):
            return self._magnification_bias

        if self.luminosity_variable == 'luminosity':
            prefactor = 2/5 * 1/np.log(10)
        elif self.luminosity_variable == 'magnitude':
            prefactor = 1/np.log(10)

        self._magnification_bias = lambda z: prefactor \
            * self.luminosity_function(self.luminosity_threshold(z), z) \
            / self.comoving_number_density(z)

        return np.vectorize(self._magnification_bias)

    @staticmethod
    def _alias(brightness_variable):

        try:
            if brightness_variable.lower().startswith('f'):
                brightness_variable = 'flux'
            elif brightness_variable.lower().startswith('l'):
                brightness_variable = 'luminosity'
            elif brightness_variable.lower().startswith('m'):
                brightness_variable = 'magnitude'
            else:
                raise ValueError(
                    "Unrecognised brightness variable: '{}'. "
                    .format(brightness_variable)
                )
        except AttributeError:
            raise TypeError("Brightness variable must be a string. ")
        else:
            return brightness_variable
