r"""
Luminosity function modeller (:mod:`~horizonground.lumfunc_modeller`)
===========================================================================

Provide models of the redshift-dependent tracer luminosity function
:math:`\Phi(m, z)` (for absolute magnitude :math:`m`) or
:math:`\Phi(\lg{L}, z)` (for base-10 logarithm of the intrinsic
luminosity :math:`L`), from which the comoving number density below/above
some brightness threshold :math:`\bar{m}` or :math:`\bar{L}`,

.. math::

    \bar{n}(z; <\!\bar{m}) = \int_{-\infty}^{\bar{m}}
        \operatorname{d}\!m\, \Phi(m, z)
    \quad \mathrm{or} \quad
    \bar{n}(z; >\!\lg\bar{L}) = \int^{\infty}_{\lg\bar{L}}
        \operatorname{d}\!\lg{L}\, \Phi(\lg{L}, z) \,,

can be predicted.  The corresponding evolution bias is

.. math::

    b_\mathrm{e}(z) = - (1 + z)
        \frac{\partial \ln\bar{n}(z)}{\partial z}

and magnification bias is

.. math::

    s(z) = \frac{1}{\ln10}
        \frac{\Phi(\bar{m},z)}{\bar{n}(z; <\!\bar{m})}
    \quad \mathrm{or} \quad
    s(z) = \frac{2}{5\ln10}
        \frac{\Phi(\lg\bar{L},z)}{\bar{n}(z; >\!\lg\bar{L})}
    \,,

We also offer a simple :math:`K`-correction formula taken from ref. [1]_.

.. [1] Croom S. M. et al., 2009. **MNRAS** 399(4), 1755--1772.
   [arXiv: `0907.2727 <https://arxiv.org/abs/0907.2727>`_]


.. autosummary::

    LumFuncModeller
    konstante_correction


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

where :math:`m` is the absolute magnitude, :math:`z` is the redshift, and
the slope parameters :math:`\alpha, \beta` describe the power laws on the
bright and faint ends respectively, but their values differ below and
above the pivot redshift :math:`z_\mathrm{p}`.  :math:`m_\ast` is the
break magnitude at which the luminosity function evaluates to
:math:`\Phi_\ast`,

.. math::

    m_\ast(z) = m_\ast(z_\mathrm{p}) - \frac{5}{2} \left[
        k_1 (z - z_\mathrm{p}) + k_2 (z - z_\mathrm{p})^2
    \right] \,,

where :math:`k_1, k_2` are the redshift-evolution parameters whose values
also differ below and above the pivot redshift [2]_.

This is a parametric model with 10 parameters: :math:`\lg\Phi_\ast`,
:math:`m_\ast(z_\mathrm{p})`, :math:`(\alpha, \beta, k_1, k_2)_\mathrm{l}`
for :math:`z < z_\mathrm{p}` and :math:`(\alpha, \beta, k_1,
k_2)_\mathrm{h}` for :math:`z > z_\mathrm{p}`.   Due to the exchange
symmetry between :math:`\alpha` and :math:`\beta` in the double power law,
the model constraint :math:`\alpha < \beta` is imposed for both above
and below the pivot redshift.


**Hybrid evolution model (PLE+LEDE)**

This model is the same as the PLE model below the pivot redshift, but
above that the luminosity evolution--density evolution (LEDE) model is
adopted where the luminosity function normalisation, break magnitude and
bright-end power law index have different redshift evolutions

.. math::

    \begin{align*}
        \lg\Phi_\ast &= \lg\Phi_\ast(z_\mathrm{p}) \
            + c_{1\mathrm{a}} (z - z_\mathrm{p})
            + c_{1\mathrm{b}} (z - z_\mathrm{p})^2 \,, \\
        m_\ast(z) &= m_\ast(z_\mathrm{p}) + c_2 (z - z_\mathrm{p}) \,, \\
        \alpha(z) &= \alpha(z_\mathrm{p}) + c_3 (z - z_\mathrm{p}) \,,
    \end{align*}

and continuity across redshift is imposed by requiring the same
:math:`\lg\Phi_\ast(z_\mathrm{p})` and :math:`m_\ast(z_\mathrm{p})` for
two models [2]_.

The hybrid model still has 10 overall parameters: the PLE model retains 6
low-redshift parameters and the high-redshift LEDE model has 8 parameters,
with the substitutions of :math:`\lg\Phi_\ast(0)` for :math:`\lg\Phi_\ast`,
:math:`m_\ast(0)` for :math:`m_\ast(z_\mathrm{p})`, :math:`c_2` for
:math:`k_1, k_2` and the addition of
:math:`c_{1\mathrm{a}}, c_{1\mathrm{b}}` and :math:`c_3`.  As before, due
to the exchange symmetry between :math:`\alpha` and :math:`\beta`, the
low-redshift PLE model constraint :math:`\alpha < \beta` is imposed.

.. [2] Palanque-Delabrouille N. et al., 2016. **A&A** 587, A41.
   [arXiv: `1509.05607 <https://arxiv.org/abs/1509.05607>`_]


.. autosummary::

    quasar_PLE_lumfunc
    quasar_PLE_model_constraint
    quasar_hybrid_lumfunc
    quasar_hybrid_model_constraint


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
                    \quad z \leqslant z_\mathrm{b} \,; \\
                \Phi_{\ast0} (1 + z_\mathrm{b})^{2\epsilon}
                    (1 + z)^{-\epsilon} \,, \quad z > z_\mathrm{b} \,,
            \end{cases} \\
        y(z) &= \frac{L}{L_{\ast0}} (1 + z)^{-\delta} \,,
    \end{align*}

are the redshift-dependent characteristic comoving number density and
relative luminosity of the H |alpha| -emitters, :math:`\epsilon, \delta`
are the redshift-evolution indices, and :math:`z_\mathrm{b}` is the break
magnitude [3]_.

This is a parametric model with 6 parameters: :math:`\alpha, \epsilon,
\delta`, :math:`z_\mathrm{b}`, :math:`m_{\ast0}` and :math:`\Phi_{\ast0}`.

.. [3] Pozzetti L. et al., 2016. **A&A** 590, A3.
   [arXiv: `1603.01453 <https://arxiv.org/abs/1603.01453>`_]


.. autosummary::

    alpha_emitter_schechter_lumfunc

|

.. |alpha| unicode:: U+03B1
    :trim:

"""
from inspect import signature

import numpy as np
from astropy import units
from scipy.integrate import quad
from scipy.misc import derivative


def konstante_correction(redshift, normalisation_redshift=2., index=-0.5):
    r"""Subtractive Konstante correction (*K*-correction) to apparent
    magnitude for correcting bandpass redshifting.

    Notes
    -----
    This implements the subtractive *K*-correction :math:`K(z) - K(z_\ast)`
    normalised to redshift :math:`z_\ast`, where

    .. math::

        K(z) = - \frac{5}{2} (1 + \alpha_\nu) \lg(1 + z)

    and :math:`\alpha_\nu` is the power-law index.

    Parameters
    ----------
    redshift : float
        Redshift.
    normalisation_redshift : float, optional
        Normalisation redshift (default is 2.).
    index : float, optional
        :math:`K`-correction power-law index (default is -0.5).

    Returns
    -------
    correction_magnitude : float
        The magnitude correction that needs to be subtracted from the
        observed magnitude.

    """
    return - 5./2. * (1 + index) \
        * np.log10((1 + redshift) / (1 + normalisation_redshift))


def quasar_PLE_lumfunc(magnitude, redshift, *, base10_log=True,
                       redshift_pivot=2.2, **model_parameters):
    r"""Evaluate the pure luminosity evolution (PLE) model for the quasar
    luminosity function at the given absolute magnitude and redshift.

    Parameters
    ----------
    magnitude : float
        Quasar absolute magnitude.
    redshift : float
        Quasar redshift.
    base10_log : bool, optional
        If `True` (default), return the base-10 logarithmic value.
    redshift_pivot : float, optional
        Pivot redshift.
    **model_parameters
        PLE model parameters.  Must be passed with the following
        parameter names:
        ``r'\lg\Phi_\ast'``, ``r'm_\ast(z_\mathrm{p})'``,
        ``r'\alpha_\mathrm{{l}}'``, ``r'\beta_\mathrm{{l}}'``,
        ``r'k_{{1\mathrm{{l}}}}'``, ``r'k_{{2\mathrm{{l}}}}'``,
        ``r'\alpha_\mathrm{{h}}'``, ``r'\beta_\mathrm{{h}}'``,
        ``r'k_{{1\mathrm{{h}}}}'``, ``r'k_{{2\mathrm{{h}}}}'``.

    Returns
    -------
    lumfunc_value : float
        Predicted qausar luminosity function value (in inverse cubic Mpc
        per unit magnitude).  Base-10 logarithmic value is returned if
        `base10_log` is `True`.

    Notes
    -----
    This function is not vectorised.  Please use `numpy.vectorize` or
    `functools` if you need vectorisation.

    """
    if not isinstance(model_parameters, dict):
        raise TypeError("Model parameters must be passed as a dictionary.")

    m, z, z_p = magnitude, redshift, redshift_pivot

    # Determine the redshift end for setting parameters.
    subscript = r'\mathrm{{{}}}'.format('l') if z <= z_p \
        else r'\mathrm{{{}}}'.format('h')

    alpha = model_parameters[r'\alpha_{}'.format(subscript)]
    beta = model_parameters[r'\beta_{}'.format(subscript)]
    k_1 = model_parameters[r'k_{{1{}}}'.format(subscript)]
    k_2 = model_parameters[r'k_{{2{}}}'.format(subscript)]

    lg_Phi_star = model_parameters[r'\lg\Phi_\ast']
    m_star_at_z_p = model_parameters[r'm_\ast(z_\mathrm{p})']

    # Evaluate the model prediction.
    magnitude_deviation_exponent = \
        (m - m_star_at_z_p) + 2.5 * (k_1 * (z - z_p) + k_2 * (z - z_p) ** 2)

    ln_faint_power_law = \
        np.log(10) * (0.4 * (alpha + 1) * magnitude_deviation_exponent)
    ln_bright_power_law = \
        np.log(10) * (0.4 * (beta + 1) * magnitude_deviation_exponent)

    ln_denominator = np.logaddexp(ln_faint_power_law, ln_bright_power_law)

    if base10_log:
        lumfunc_value = lg_Phi_star - ln_denominator / np.log(10)
    else:
        Phi_star = 10 ** lg_Phi_star
        lumfunc_value = Phi_star / np.exp(ln_denominator)

    return lumfunc_value


def quasar_PLE_model_constraint(**model_parameters):
    """Check whether the pure luminosity evolution (PLE) model constraint
    is satisfied.

    Parameters
    ----------
    **model_parameters
        PLE model parameters.  See :func:`quasar_PLE_lumfunc` for required
        parameters.

    Returns
    -------
    bool
        Whether or not the model constraint is satisfied.

    """
    return (
        model_parameters[r'\alpha_\mathrm{l}']
        < model_parameters[r'\beta_\mathrm{l}']
    ) and (
        model_parameters[r'\alpha_\mathrm{h}']
        < model_parameters[r'\beta_\mathrm{h}']
    )


def quasar_hybrid_lumfunc(magnitude, redshift, *, base10_log=True,
                          redshift_pivot=2.2, **model_parameters):
    r"""Evaluate the hybrid model (pure luminosity evolution and luminosity
    evolution--density evolution, 'PLE+LEDE') for the quasar luminosity
    function at the given absolute magnitude and redshift.

    Parameters
    ----------
    magnitude : float
        Quasar absolute magnitude.
    redshift : float
        Quasar redshift.
    base10_log : bool, optional
        If `True` (default), return the base-10 logarithmic value.
    redshift_pivot : float, optional
        Pivot redshift.
    **model_parameters
        Hybrid model parameters.  Must be passed with the following
        parameter names: ``r'\lg\Phi_\ast(0)'``, ``r'm_\ast(0)'``,
        ``r'\alpha'``, ``r'\beta'``, ``r'k_1'``, ``r'k_2'``,
        ``r'c_{{1\mathrm{{a}}}}'``, ``r'c_{{1\mathrm{{b}}}}'``,
        ``r'c_2'``, ``r'k_3'``.

    Returns
    -------
    lumfunc_value : float
        Predicted qausar luminosity function value (in inverse cubic Mpc
        per unit magnitude).  Base-10 logarithmic value is returned if
        `base10_log` is `True`.

    Notes
    -----
    This function is not vectorised.  Please use `numpy.vectorize` or
    `functools` if you need vectorisation.

    """
    if not isinstance(model_parameters, dict):
        raise TypeError("Model parameters must be passed as a dictionary.")

    m, z, z_p = magnitude, redshift, redshift_pivot

    # Low redshift PLE model.
    lg_Phi_star_0 = model_parameters[r'\lg\Phi_\ast(0)']
    m_star_0 = model_parameters[r'm_\ast(0)']
    alpha = model_parameters[r'\alpha']
    beta = model_parameters[r'\beta']
    k_1 = model_parameters[r'k_1']
    k_2 = model_parameters[r'k_2']

    if z <= z_p:

        magnitude_deviation_exponent = \
            (m - m_star_0) + 2.5 * (k_1 * z + k_2 * z ** 2)

        ln_faint_power_law = \
            np.log(10) * (0.4 * (alpha + 1) * magnitude_deviation_exponent)
        ln_bright_power_law = \
            np.log(10) * (0.4 * (beta + 1) * magnitude_deviation_exponent)

        ln_denominator = np.logaddexp(ln_faint_power_law, ln_bright_power_law)

        if base10_log:
            lumfunc_value = lg_Phi_star_0 - ln_denominator / np.log(10)
        else:
            Phi_star = 10 ** lg_Phi_star_0
            lumfunc_value = Phi_star / np.exp(ln_denominator)

        return lumfunc_value

    # High redshift LEDE model.
    c_1a = model_parameters[r'c_{1\mathrm{a}}']
    c_1b = model_parameters[r'c_{1\mathrm{b}}']
    c_2 = model_parameters[r'c_2']
    c_3 = model_parameters[r'c_3']

    lg_Phi_star_at_z_p = lg_Phi_star_0
    m_star_at_z_p = m_star_0 - 2.5 * (k_1 * z_p + k_2 * z_p ** 2)

    lg_Phi_star = lg_Phi_star_at_z_p + c_1a * (z - z_p) + c_1b * (z - z_p) ** 2
    magnitude_deviation_exponent = (m - m_star_at_z_p) + c_2 * (z - z_p)
    alpha = alpha + c_3 * (z - z_p)

    ln_faint_power_law = \
        np.log(10) * (0.4 * (alpha + 1) * magnitude_deviation_exponent)
    ln_bright_power_law = \
        np.log(10) * (0.4 * (beta + 1) * magnitude_deviation_exponent)

    ln_denominator = np.logaddexp(ln_faint_power_law, ln_bright_power_law)

    if base10_log:
        lumfunc_value = lg_Phi_star - ln_denominator / np.log(10)
    else:
        Phi_star = 10 ** lg_Phi_star
        lumfunc_value = Phi_star / np.exp(ln_denominator)

    return lumfunc_value


def quasar_hybrid_model_constraint(**model_parameters):
    r"""Check whether the hybrid model constraint is satisfied.

    Parameters
    ----------
    **model_parameters
        Hybrid model parameters.  See :func:`quasar_hybrid_lumfunc` for
        required parameters.

    Returns
    -------
    bool
        Whether or not the model constraint is satisfied.

    """
    return model_parameters[r'\alpha'] < model_parameters[r'\beta']


def alpha_emitter_schechter_lumfunc(luminosity, redshift, *, base10_log=True,
                                    **model_parameters):
    r"""Evaluate the Schechter model for the H |alpha| -emitter
    luminosity function at the given intrinsic luminosity and redshift.

    Parameters
    ----------
    luminosity : float
        H |alpha| -emitter luminosity in dex (base-10 logarithm).
    redshift : float
        H |alpha| -emitter redshift.
    base10_log : bool, optional
        If `True` (default), return the base-10 logarithmic value.
    **model_parameters
        Schechter model parameters.  Must be passed with the following
        parameter names: ``r'\lg\Phi_{\ast0}'``, ``r'\lg{L_{\ast0}}'``,
        ``r'z_\mathrm{b}'``, ``r'\alpha'``, ``r'\delta'``, ``r'\epsilon'``.

    Returns
    -------
    lumfunc_value : float
        Predicted H |alpha| -emitter luminosity function value (in inverse
        cubic Mpc per flux dex).  Base-10 logarithmic value is returned if
        `base10_log` is `True`.

    Notes
    -----
    This function is not vectorised.  Please use `numpy.vectorize` or
    `functools` if you need vectorisation.

    """
    lg_L, z = luminosity, redshift

    lg_Phi_star0 = model_parameters[r'\lg\Phi_{\ast0}']
    lg_L_star0 = model_parameters[r'\lg{L_{\ast0}}']
    z_b = model_parameters[r'z_\mathrm{b}']
    alpha = model_parameters[r'\alpha']
    delta = model_parameters[r'\delta']
    epsilon = model_parameters[r'\epsilon']

    # Evaluate the model prediction.
    if z <= z_b:
        lg_Phi_star = lg_Phi_star0 + epsilon * np.log10(1 + z)
    else:
        lg_Phi_star = lg_Phi_star0 \
            + 2 * epsilon * np.log10(1 + z_b) \
            - epsilon * np.log10(1 + z)

    lg_y = lg_L - lg_L_star0 - delta * np.log10(1 + z)

    if base10_log:
        lumfunc_value = np.log10(np.log(10)) + lg_Phi_star \
            + (alpha + 1) * lg_y - 10 ** lg_y * np.log10(np.e)
    else:
        lumfunc_value = np.log(10) * 10 ** lg_Phi_star \
            * 10 ** (lg_y * (alpha + 1)) * np.exp(- 10 ** lg_y)

    return lumfunc_value


class LumFuncModeller:
    r"""Luminosity function modeller predicting the comoving number
    density, evolution bias and magnification bias for a given brightness
    threshold.

    Parameters
    ----------
    model_lumfunc : callable
        A parametric luminosity function model as a function of brightness
        and redshift (in that order) in units of inverse cubic Mpc.  If it
        returns the luminosity function value in base-10 logarithm,
        `exponentiation` should be set to `True`; if it accepts a boolean
        parameter 'base10_log', this will be overriden to `False`.
    model_parameters : dict
        Model parameters passed to `model_lumfunc` as keyword arguments.
    brightness_variable : {'luminosity', 'magnitude'}, str
        Brightness variable of `model_lumfunc`, either 'luminosity' (in
        dex) or 'magnitude'.
    threshold_value : float
        Brightness threshold value for `brightness_variable`.  If
        `brightness_variable` is 'luminosity', this is interpreted as a
        flux limit and converted to an intrinsic luminosity value at the
        tracer redshift using the luminosity distance.
    normalise_threshold : bool, optional
        If `True` (default is `False`), `threshold_value` is converted
        from a flux limit to a luminosity value at `normalisation_redshift`
        using the luminosity distance if `brightness_variable` is
        'luminosity', or converted from an apparent magnitude limit to
        an absolute magnitude value at `normalisation_redshift` using the
        distance modulus if `brightness_variable` is 'magnitude'.
    normalisation_redshift : float or None, optional
        If not `None` (default) and `normalise_threshold` is `True`, this
        redshift is used convert flux or apparent magnitude limit to an
        intrinsic luminosity or absolute magnitude threshold.
    cosmology : :class:`astropy.cosmology.core.Cosmology` or None, optional
        Background cosmological model used to calculate the luminosity
        distance or distance modulus (default is `None`).
    exponentiation : bool, optional
        If `True` (default is `False`), this assumes `model_lumfunc` only
        returns the base-10 logarithmic luminosity function and
        exponentiate its returned value to a power of 10.

    Attributes
    ----------
    luminosity_function : callable
        Luminosity function of brightness and redshift variables only
        (in that order) (in inverse cubic :math:`\mathrm{Mpc}/h` per unit
        luminosity).
    brightness_threshold : callable
        Intrinsic luminosity or absoloute magnitude threshold, depending
        on the input argument of `brightness_variable`, as a function
        of redshift.
    cosmology : :class:`astropy.cosmology.core.Cosmology`
        Background cosmological model.
    attrs : dict
        Model parameters and luminosity variables stored in a dictionary.

    Notes
    -----
    :attr:`luminosity_function` is a vectorised function.

    """

    luminosity_bound = {
        'luminosity': 100.,
        'magnitude': -50.,
    }
    r"""float: Finite luminosity upper bound for numerical integration.

    If :attr:`brightness_variable` is 'luminosity', the bound is given as
    a base-10 logarithmic flux value in erg/s; else if 'magnitude',
    the bound is dimensionless.

    """

    redshift_stepsize = 0.001
    r"""float: Redshift step size for numerical differentiation.

    """

    def __init__(self, model_lumfunc, model_parameters, brightness_variable,
                 threshold_value, normalise_threshold=False,
                 normalisation_redshift=None, cosmology=None,
                 exponentiation=False):

        self.attrs = {
            'model_parameters': model_parameters,
            'brightness_variable': brightness_variable,
        }

        self.cosmology = cosmology

        if 'base10_log' in signature(model_lumfunc).parameters:
            model_parameters.update({'base10_log': False})
            # pylint: disable=unnecessary-lambda
            self.luminosity_function = np.vectorize(
                lambda lum, z: model_lumfunc(lum, z, **model_parameters)
            )
        elif exponentiation:
            self.luminosity_function = np.vectorize(
                lambda lum, z: 10 ** model_lumfunc(lum, z, **model_parameters)
            )
        else:
            # pylint: disable=unnecessary-lambda
            self.luminosity_function = np.vectorize(
                lambda lum, z: model_lumfunc(lum, z, **model_parameters)
            )

        if brightness_variable == 'luminosity':
            # pylint: disable=no-member
            self.brightness_threshold = lambda z: np.log10(
                4 * np.pi * threshold_value
                * self.cosmology.luminosity_distance(z).to(units.cm).value ** 2
            )
            self._brightness_threshold = \
                self.brightness_threshold(normalisation_redshift)
        elif brightness_variable == 'magnitude':
            self.brightness_threshold = lambda z: \
                threshold_value - self.cosmology.distmod(z).value \
                if normalise_threshold else threshold_value
            self._brightness_threshold = \
                self.brightness_threshold(normalisation_redshift) \
                if normalise_threshold else threshold_value

    @classmethod
    def from_parameter_file(cls, parameter_file, **kwargs):
        """Instantiate the modeller class with model parameter values
        loaded from a file.

        Parameters
        ----------
        parameter_file : *str or* :class:`pathlib.Path`
            Path of the model parameter file.
        **kwargs
            Parameters other than `model_parameters` passed to the
            modeller class as keyword arguments.

        """
        with open(parameter_file, 'r') as pfile:
            parameters = tuple(
                map(
                    lambda var_name: var_name.strip(),
                    pfile.readline().strip("#").strip("\n").split(",")
                )
            )
            estimates = tuple(map(float, pfile.readline().split(",")))

        model_parameters = dict(zip(parameters, estimates))

        return cls(model_parameters=model_parameters, **kwargs)

    def comoving_number_density(self, redshift):
        r"""Return the comoving number density :math:`\bar{n}(z)`
        above the luminosity threshold at a given redshift.

        Parameters
        ----------
        redshift : float
            Redshift.

        Returns
        -------
        float
            Comoving number density (in inverse cubic
            :math:`\mathrm{Mpc}/h`).

        """
        _comoving_number_density = np.abs(
            quad(
                self.luminosity_function,
                self._brightness_threshold,
                self.luminosity_bound[self.attrs['brightness_variable']],
                args=(redshift,)
            )[0]
        ) / self.cosmology.h ** 3

        return _comoving_number_density

    def evolution_bias(self, redshift):
        r"""Return the evolution bias :math:`b_\mathrm{e}(z)` at a
        given redshift.

        Parameters
        ----------
        redshift : float
            Redshift.

        Returns
        -------
        float
            Evolution bias.

        """
        _evolution_bias = - (1 + redshift) * derivative(
            self.comoving_number_density, redshift, dx=self.redshift_stepsize
        ) / self.comoving_number_density(redshift)

        return _evolution_bias

    def magnification_bias(self, redshift):
        r"""Return the magnification bias :math:`s(z)` at a given redshift.

        Parameters
        ----------
        redshift : float
            Redshift.

        Returns
        -------
        float
            Magnification bias.

        """
        if self.attrs['brightness_variable'] == 'luminosity':
            prefactor = 2./5. / np.log(10)
        elif self.attrs['brightness_variable'] == 'magnitude':
            prefactor = 1 / np.log(10)

        _magnification_bias = prefactor * self.luminosity_function(
            self._brightness_threshold, redshift
        ) / self.comoving_number_density(redshift)

        return _magnification_bias
