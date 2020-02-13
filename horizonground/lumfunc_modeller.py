r"""
Luminosity Function Modeller (:mod:`~horizonground.lumfunc_modeller`)
===========================================================================

Model the redshift-dependent tracer luminosity function :math:`\Phi(m, z)`
(for suitably normalised magnitude :math:`m`) or :math:`\Phi(\lg{L}, z)`
(for base-10 logarithmic luminosity), predict the comoving number density

.. math::

    \bar{n}(z;\bar{m}) = \int_{-\infty}^{\bar{m}}
        \operatorname{d}\!m \Phi(m, z)
    \quad \textrm{or} \quad
    \bar{n}(z;\lg\bar{L}) = \int^{\infty}_{\lg\bar{L}}
        \operatorname{d}\!\lg{L} \Phi(\lg{L}, z) \,.

and derive the corresponding evolution bias

.. math::

    f_\textrm{ev}(z) = - (1 + z)
        \frac{\partial \ln\bar{n}(z)}{\partial z}

and magnification bias

.. math::

    s(z) = \frac{1}{\ln10} \frac{\Phi(\bar{m},z)}{\bar{n}(z;\bar{m})}
    \quad \textrm{or} \quad
    s(z) = \frac{2}{5\ln10} \frac{\Phi(\lg\bar{L},z)}{\bar{n}(z;\bar{L})}
    \,.

.. autosummary::

    LumFuncModeller


Quasar (QSO) luminosity function
---------------------------------------------------------------------------

**Pure luminosity evolution model (PLE)**

The quasar luminosity function in the pure luminosity evolution (PLE)
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

**Hybrid evolution model (PLE+LEDE)**

.. todo:: Not implemented.

.. autosummary::

    quasar_PLE_model
    quasar_hybrid_model


H |alpha| -emitter luminosity function
---------------------------------------------------------------------------

**Schechter function model**

The H |alpha| -emitter luminosity function in the Schechter function model
takes the form of a gamma function

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

.. autosummary::

    alpha_emitter_schechter_model

|

.. |alpha| unicode:: U+03B1
    :trim:

"""
from __future__ import division

import numpy as np
from astropy import units
from scipy.integrate import quad
from scipy.misc import derivative


def quasar_PLE_model(magnitude, redshift, redshift_pivot=2.2, base10_log=True,
                     **model_parameters):
    """Evaluate the pure luminosity evolution (PLE) model for the quasar
    luminosity function at the given magnitude and redshift.

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
    base10_log : bool, optional
        If `True` (default), return the base-10 logarithmic value.
    **model_parameters
        PLE model parameters.

    Returns
    -------
    comoving_density : float :class:`numpy.ndarray`
        Predicted qausar comoving number density per unit magnitude (in
        inverse cubic Mpc).  If `base10_log` is `True`, the base-10
        logarithmic value is returned.

    """
    # Re-definitions.
    M_g, z, z_p = magnitude, redshift, redshift_pivot

    # Determine the redshift range.
    if z <= z_p:
        subscript = r'\textrm{{{}}}'.format('l')
    else:
        subscript = r'\textrm{{{}}}'.format('h')

    # Set parameters.
    M_g_star_p = model_parameters[r'M_{g\ast}(z_\textrm{p})']

    alpha = model_parameters[r'\alpha_{}'.format(subscript)]
    beta = model_parameters[r'\beta_{}'.format(subscript)]

    k_1 = model_parameters[r'k_{{1{}}}'.format(subscript)]
    k_2 = model_parameters[r'k_{{2{}}}'.format(subscript)]

    # Evaluate the model prediction.
    exponent_magnitude_factor = M_g - M_g_star_p \
         + 2.5*(k_1 * (z - z_p) + k_2 * (z - z_p)**2)

    try:
        faint_power_law = 10 ** (0.4*(alpha + 1) * exponent_magnitude_factor)
    except OverflowError:
        if (alpha + 1) * exponent_magnitude_factor > 0.:
            faint_power_law = np.inf
        else:
            faint_power_law = 0.
    try:
        bright_power_law = 10 ** (0.4*(beta + 1) * exponent_magnitude_factor)
    except OverflowError:
        if (beta + 1) * exponent_magnitude_factor > 0.:
            bright_power_law = np.inf
        else:
            bright_power_law = 0.

    if base10_log:
        lg_Phi_star = model_parameters[r'\lg\Phi_\ast']
        comoving_density = lg_Phi_star \
            - np.log10(faint_power_law + bright_power_law)
    else:
        Phi_star = 10**model_parameters[r'\lg\Phi_\ast']
        comoving_density = Phi_star / (faint_power_law + bright_power_law)

    return comoving_density


def quasar_hybrid_model(magnitude, redshift, redshift_pivot=2.2,
                        base10_log=True, **model_parameters):
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
    base10_log : bool, optional
        If `True` (default), return the base-10 logarithmic value.
    **model_parameters
        'PLE+LEDE' hybrid model parameters.

    Returns
    -------
    comoving_density : float :class:`numpy.ndarray`
        Predicted qausar comoving number density per unit magnitude (in
        inverse cubic Mpc).  If `base10_log` is `True`, the base-10
        logarithmic value is returned.

    """
    raise NotImplementedError


def alpha_emitter_schechter_model(lg_luminosity, redshift, base10_log=True,
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
    base10_log : bool, optional
        If `True` (default), return the base-10 logarithmic value.
    **model_parameters
        Schechter model parameters.

    Returns
    -------
    comoving_density : float :class:`numpy.ndarray`
        Predicted emitter comoving number density per unit base-10
        logarithmic luminosity (in inverse cubic Mpc).  If `base10_log` is
        `True`, the base-10 logarithmic value is returned.

    """
    # Re-definitions.
    z = redshift

    # Set parameters.
    z_b = model_parameters[r'z_\textrm{b}']

    alpha = model_parameters[r'\alpha']
    delta = model_parameters[r'\delta']
    epsilon = model_parameters[r'\epsilon']

    # Evaluate the model prediction.
    if base10_log:
        lg_Phi_star0 = model_parameters[r'\lg\Phi_{\ast0}']
    else:
        Phi_star0 = 10**model_parameters[r'\lg\Phi_{\ast0}']

    if z <= z_b:
        if base10_log:
            lg_Phi_star = lg_Phi_star0 + epsilon * np.log10(1 + z)
        else:
            Phi_star = Phi_star0 * (1 + z)**epsilon
    else:
        if base10_log:
            lg_Phi_star = lg_Phi_star0 \
                + 2*epsilon * np.log10(1 + z_b) \
                - epsilon * np.log10(1 + z)
        else:
            Phi_star = Phi_star0 * (1 + z_b)**(2*epsilon) / (1 + z)**epsilon

    if base10_log:
        lg_y0 = lg_luminosity - model_parameters[r'\lg{L_{\ast0}}']
        lg_y = lg_y0 - delta * np.log10(1 + z)
        comoving_density = np.log10(np.log(10)) + lg_Phi_star \
            + (alpha + 1) * lg_y - np.log10(np.e) * 10**lg_y
    else:
        y0 = 10**(lg_luminosity - model_parameters[r'\lg{L_{\ast0}}'])
        y = y0 / (1 + z)**delta
        comoving_density = np.log(10) * Phi_star * y**(alpha + 1) * np.exp(-y)

    return comoving_density


class LumFuncModeller:
    r"""Luminosity function modeller predicting the comoving number
    density, evolution bias and magnification bias for a given brightness
    threshold.

    Parameters
    ----------
    lumfunc_model : callable
        A parametric model for the luminosity function of luminosity or
        magnitude and redshift variables (in that order).
    brightness_variable : {'luminosity', 'magnitude'}, str
        Luminosity variable, either ``'luminosity'`` or ``'magnitude'``.
    threshold_value : float
        Brightness threshold value.
    threshold_variable : {'flux', 'luminosity', 'magnitude'}, str
        Brightness threshold variable, one of ``'flux'``, ``'luminosity'``
        or ``'magnitude'``.  If ``'flux'``, `threshold_value` will be
        converted into ``'luminosity'`` threshold value.
    cosmology : :class:`astropy.cosmology.Cosmology`
        Background cosmological model.
    **model_parameters
        Model parameters to be passed to `lumfunc_model` as.

    Attributes
    ----------
    luminosity_function : callable
        Luminosity function of luminosity or magnitude and redshift
        variables only (in that order) (in inverse cubic Mpc/:math:`h`
        per unit brightness).
    brightness_variable : {'luminosity', 'magnitude'}, str
        Brightness variable, either ``'luminosity'`` or ``'magnitude'``.
    brightness_threshold : callable
        Brightness threshold in :attr:`brightness_variable` as a function
        of redshift.
    model_parameters : dict
        Model parameters.
    cosmology : :class:`astropy.cosmology.Cosmology`
        Background cosmological model.

    """

    # HINT: Instead of ``np.inf`` to prevent arithmetic overflow.
    brightness_bound = {
        'luminosity': 100.,
        'magnitude': -40.,
    }
    r"""float: Finite brightness bound.

    If :attr:`brightness_variable` is ``'luminosity'``, it is the base-10
    logarithmic value in erg/s; else if it is ``'magnitude'`` and
    dimensionless.

    """

    redshift_stepsize = 0.01
    r"""float: Redshift step size for numerical computations.

    """

    def __init__(self, lumfunc_model, brightness_variable, threshold_value,
                 threshold_variable, cosmology, **model_parameters):

        self.model_parameters = model_parameters
        self.cosmology = cosmology

        self.luminosity_function = np.vectorize(
            lambda lum, z: lumfunc_model(lum, z, **self.model_parameters)
        )

        # HINT: Default values agrees with luminosity function models.
        self._lg_conversion = self.model_parameters.get('base10_log', True)

        self._threshold_variable = self._alias(threshold_variable)
        if self._threshold_variable == 'flux':
            self.brightness_threshold = lambda z: np.log10(
                4*np.pi
                * threshold_value
                * self.cosmology.luminosity_distance(z).to(units.cm).value**2
            )
            self._threshold_variable = 'luminosity'
        else:
            self.brightness_threshold = lambda z: threshold_value

        self.brightness_variable = self._alias(brightness_variable)
        if self._threshold_variable != self.brightness_variable:
            raise ValueError(
                "`threshold_variable` '{}' does not match "
                "`brightness_variable` '{}'. "
                .format(self._threshold_variable, self.brightness_variable)
            )

        self._comoving_number_density = None
        self._evolution_bias = None
        self._magnification_bias = None

    @classmethod
    def from_parameters_file(cls, parameter_file, lumfunc_model,
                             brightness_variable, threshold_value,
                             threshold_variable, cosmology=None,
                             **model_parameters):
        """Instantiate the modeller by loading model parameter values from
        a file.

        Parameters
        ----------
        parameter_file : str
            Path of the model parameter file.
        lumfunc_model : callable
            A parametric model for the luminosity function of luminosity or
            magnitude and redshift variables accepting additional
            parameters (in that order).
        brightness_variable : {'luminosity', 'magnitude'}, str
            Luminosity variable, either ``'luminosity'`` or
            ``'magnitude'``.
        threshold_value : float
            Luminosity threshold value.
        threshold_variable : {'flux', 'luminosity', 'magnitude'}, str
            Luminosity threshold variable, one of ``'flux'``,
            ``'luminosity'`` or ``'magnitude'``.  If ``'flux'``,
            `threshold_value` will be converted into ``'luminosity'``
            threshold value.
        cosmology : :class:`astropy.cosmology.Cosmology`
            Background cosmological model.
    **model_parameters
        Model parameters to be passed to `lumfunc_model` as.

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

        model_parameters.update(dict(zip(parameters, estimates)))

        return cls(
            lumfunc_model, brightness_variable,
            threshold_value, threshold_variable,
            cosmology=cosmology, **model_parameters
        )

    @property
    def comoving_number_density(self):
        r"""Comoving number density.

        Returns
        -------
        callable
            Comoving number density (in inverse cubic Mpc/:math:`h`)
            as a function of redshift.

        """
        if callable(self._comoving_number_density):
            return self._comoving_number_density

        if self._lg_conversion:
            def _luminosity_function(*args, **kwargs):
                return 10**self.luminosity_function(*args, **kwargs)
        else:
            _luminosity_function = self.luminosity_function

        self._comoving_number_density = lambda z: np.abs(
            quad(
                _luminosity_function,
                self.brightness_threshold(z),  # faint end
                self.brightness_bound[self.brightness_variable],  # bright end
                args=(z,)
            )[0]
        ) / self.cosmology.h**3

        return np.vectorize(self._comoving_number_density)

    @property
    def evolution_bias(self):
        r"""Evolution bias.

        Returns
        -------
        callable
            Evolution bias as a function of redshift.

        """
        if callable(self._evolution_bias):
            return self._evolution_bias

        self._evolution_bias = lambda z: - (1 + z) * derivative(
            self.comoving_number_density, z, dx=self.redshift_stepsize
        ) / self.comoving_number_density(z)

        return np.vectorize(self._evolution_bias)

    @property
    def magnification_bias(self):
        r"""Magnification bias.

        Returns
        -------
        callable
            Magnification bias as a function of redshift.

        """
        if callable(self._magnification_bias):
            return self._magnification_bias

        if self.brightness_variable == 'luminosity':
            prefactor = 2/5 * 1/np.log(10)
        elif self.brightness_variable == 'magnitude':
            prefactor = 1/np.log(10)

        self._magnification_bias = lambda z: prefactor \
            * self.luminosity_function(self.brightness_threshold(z), z) \
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
