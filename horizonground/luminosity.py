r"""
Luminosity Modeller (:mod:`luminosity`)
===========================================================================

Model luminosity function and related quantities.


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

    \Phi(m, z) = -0.4\ln10 \, \Phi_\ast(z) 10^{-0.4(\alpha+1)[m-m_\ast(z)]}
        \exp\left(- 10^{- 0.4[m-m_\ast(z)]}\right)

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
        m_\ast(z) &= m_{\ast0} - \frac{\delta}{0.4\ln10} \ln(1+z) \,,
    \end{align*}

are the redshift-dependent characteristic comoving number density and
magnitude of the H |alpha| -emitters, :math:`\epsilon, \delta` are the
redshift-evolution indices, and :math:`z_\textrm{b}` is the break
magnitude.

This is a parametric model with 6 parameters: :math:`\alpha, \epsilon,
\delta`, :math:`z_\textrm{b}`, :math:`m_{\ast0}` and :math:`\Phi_{\ast0}`.

.. |alpha| unicode:: U+03B1
    :trim:

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


def alpha_emitter_luminosity_schechter_model(magnitude, redshift,
                                             **model_parameters):
    r"""Evaluate the Schechter model for the H |alpha_| -emitter
    luminosity function at the given absolute magnitude and redshift.

    Parameters
    ----------
    magnitude : float
        Emitter magnitude.
    redshift : float
        Emitter redshift.
    **model_parameters
        Schechter model parameters.

    Returns
    -------
    comoving_density : float :class:`numpy.ndarray`
        Predicted emitter comoving number density per unit magnitude (in
        inverse cubic Mpc).


    .. |alpha_| unicode:: U+03B1
        :trim:

    """
    # Re-definitions.
    m, z = magnitude, redshift

    # Set parameters.
    Phi_star0 = model_parameters['\\lg\\Phi_{\\ast0}']
    m_star0 = - 2.5*model_parameters['\\lg{L_{\\ast0}}']

    z_b = model_parameters['z_\\textrm{b}']

    alpha = model_parameters['\\alpha']
    delta = model_parameters['\\delta']
    epsilon = model_parameters['\\epsilon']

    # Evaluate the model prediction.
    prefactor = 0.4 * np.log(10)

    if z <= z_b:
        char_density = Phi_star0 * (1 + z)**epsilon
    else:
        char_density = Phi_star0 * (1 + z_b)**(2*epsilon) / (1 + z)**epsilon

    m_star_z = m_star0 - delta / prefactor * np.log(1 + z)
    power_law = 10**(-0.4 * (alpha+1) * (m - m_star_z))
    exponential = np.exp(- 10**(-0.4 * (m - m_star_z)))

    comoving_density = - prefactor * char_density * power_law * exponential

    return comoving_density


class LuminosityFunctionModeller:
    r"""Luminosity function modeller predicting the comoving number density
    and related quantities for a given brightness threshold.

    Parameters
    ----------
    luminosity_model : callable
        A parametric model for the luminosity function with magnitude,
        redshift and additional arguments (in that order).
    threshold : float
        Brightness threshold.
    brightness_variable : {'luminosity', 'magnitude'}, str
        Brightness variable, either 'luminosity' or 'magnitude'.
    **model_parameters
        Keyword arguments to be passed to `luminosity_model`.

    Attributes
    ----------
    luminosity_function : callable
        Luminosity function of magnitude and redshift arguments only (in
        that order) (in inverse cubic Mpc).
    brightness_variable : {'luminosity', 'magnitude'}, str
        Brightness variable, either 'luminosity' or 'magnitude'.
    threshold : float
        Brightness threshold.
    model_parameters : dict
        Model parameters.

    """

    # HINT: Instead of ``np.inf`` to prevent arithmetic overflow.
    BRIGHTNESS_BOUND = {
        'magnitude': -40.,
        'luminosity': 41.5,
    }
    r"""float: Finite brightness bound.

    """

    def __init__(self, luminosity_model, threshold, brightness_variable,
                 **model_parameters):

        self.luminosity_function = np.vectorize(
            lambda lum, z: luminosity_model(lum, z, **model_parameters)
        )
        self.brightness_variable = brightness_variable
        self.threshold = threshold
        self.model_parameters = model_parameters

        self._comoving_number_density = None
        self._evolution_bias = None
        self._magnification_bias = None

    @classmethod
    def from_parameters_file(cls, luminosity_model, threshold,
                             brightness_variable, file_path):
        """Instantiate the modeller by loading model parameter values from
        a file.

        Parameters
        ----------
        luminosity_model : callable
            A parametric model for the luminosity function with brightness,
            redshift and additional arguments (in that order).
        threshold : float
            Brightness threshold.
        brightness_variable : {'luminosity', 'magnitude'}, str
            Brightness variable, either 'luminosity' or 'magnitude'.
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
            luminosity_model, threshold, brightness_variable,
            **dict(zip(parameters, estimates))
        )

    @property
    def comoving_number_density(self):
        r"""Comoving number density (in inverse cubic Mpc) as a function of
        redshift

        .. math::

            \bar{n}(z; \bar{m}) = \int_{-\infty}^{\bar{m}}
                \operatorname{d}\!m \Phi(m, z)
            \quad \textrm{or} \quad
            \bar{n}(z; \bar{L}) = \int^{\infty}_{\bar{L}}
                \operatorname{d}\!L \Phi(L, z)

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
                self.BRIGHTNESS_BOUND[self.brightness_variable],
                self.threshold,
                args=(z,)
            )[0]
        )

        return np.vectorize(self._comoving_number_density)

    @property
    def evolution_bias(self):
        r"""Evolution bias as a function of redshift

        .. math::

            f_\textrm{ev}(z) = 3 - (1 + z)
                \partial_z \ln \bar{n}(z; \bar{m})
            \quad \textrm{or} \quad
            f_\textrm{ev}(z) = 3 - (1 + z)
                \partial_z \ln \bar{n}(z; \bar{L}) \,.

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

        self._evolution_bias = lambda z: 3 - (1 + z) * derivative(
            ln_comoving_number_density, z, dx=1e-2
        )

        return np.vectorize(self._evolution_bias)

    @property
    def magnification_bias(self):
        r"""Magnification bias as a function of redshift

        .. math::

            s(z) = \left.
                \frac{\partial}{\partial{\tilde{m}}}
            \right\vert_{\bar{m}} \lg \bar{n}(z; \tilde{m})
            \quad \textrm{or} \quad
            s(z) = - \frac{2}{5} \bar{L} \left.
                \frac{\partial}{\partial{\tilde{L}}}
            \right\vert_{\bar{L}} \ln \bar{n}(z; \tilde{L}) \,.

        Returns
        -------
        callable
            Magnification bias.

        """
        if callable(self._magnification_bias):
            return self._magnification_bias

        if self.brightness_variable == 'luminosity':
            prefactor = - (2/5) * self.threshold
        elif self.brightness_variable == 'magnitude':
            prefactor = 1 / np.log(10)

        self._magnification_bias = lambda z: prefactor \
            * self.luminosity_function(self.threshold, z) \
            / self.comoving_number_density(z)

        return np.vectorize(self._magnification_bias)
