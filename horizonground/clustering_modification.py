r"""
Clustering modification (:mod:`~horizonground.clustering_modification`)
===========================================================================

Compute modifications to the Newtonian power spectrum in the
distant-observer and plane-parallel limits.


Standard Kaiser RSD model
---------------------------------------------------------------------------

Redshift-space distortions induce anisotropy in the power spectrum
multipoles

.. math::

    \begin{align*}
        P_0(k, z) &=
            \left[
                b_1(z)^2 + \frac{2}{3} b_1(z) f(z) + \frac{1}{5} f(z)^2
            \right] P_\mathrm{m}(k, z) \,, \\
        P_2(k, z) &=
            \left[
                \frac{4}{3} b_1(z) f(z) + \frac{4}{7} f(z)^2
            \right] P_\mathrm{m}(k, z) \,, \\
        P_4(k, z) &= \frac{8}{35} f(z)^2 P_\mathrm{m}(k, z) \,
    \end{align*}

where :math:`P_\mathrm{m}` is the matter power spectrum, :math:`b_1(z)` is
the linear bias and :math:`f(z)` is the linear growth rate.

.. autosummary::

    standard_kaiser_factor


Non-Gaussianity scale-dependent modifications
---------------------------------------------------------------------------

Local primordial non-Gaussianty :math:`f_\mathrm{NL}` induces scale
dependence in the linear bias,

.. math::

    b_1(z) \mapsto b_1(z) + \Delta b_k(z) \,, \quad
    \Delta b(k, z) = f_\mathrm{NL} [b_1(z) - p] \frac{A(k, z)}{k^2} \,,

where the scale-dependence kernel is

.. math::

    A(k, z) = 3 \left( \frac{H_0}{\mathrm{c}} \right)^2
        \frac{1.27 \varOmega_\mathrm{m,0} \delta_\mathrm{c}}{D(z)T(k)} \,.

Here :math:`H_0` is the Hubble parameter at the current epoch
(in km/s/Mpc), :math:`\mathrm{c}` the speed of light,
:math:`\varOmega_\mathrm{m,0}` the matter density parameter at the current
epoch, and :math:`\delta_\mathrm{c}` the critical over-density in
spherical gravitational collapse.  The growth factor :math:`D(z)` is
normalised to unity at the current epoch (thus the numerical factor 1.27),
the transfer function :math:`T(k)` is normalised to unity as
:math:`k \to 0`, and :math:`p` is a tracer-dependent parameter.

Modifications to power spectrum multipoles as a result of local primordial
non-Gaussianty are

.. math::

    \begin{align*}
        \Delta P_0(k, z) &= \left[
            \left( 2 b_1 + \frac{2}{3} f \right) \Delta b(k, z)
            + \Delta b(k, z)^2
        \right] P_\mathrm{m}(k, z) \,, \\
        \Delta P_2(k, z) &=
            \frac{4}{3} f \Delta b(k, z) P_\mathrm{m}(k, z) \,.
    \end{align*}

.. autosummary::

    scale_dependence_kernel
    non_gaussianity_correction_factor


Relativistic corrections
---------------------------------------------------------------------------

Relativistic corrections to the Newtonian clustering mode

.. math::

    \delta(\mathbf{k}, z) = \left[
        b_1(z) + f \mu^2
        + \mathrm{i} \frac{\mathcal{H}}{k} g_1(z) f(z) \mu
        + \left( \frac{\mathcal{H}}{k} \right)^2 g_2(z)
    \right] \delta_\mathrm{m}(\mathbf{k}, z) \,,

are parametrised by the redshift-dependent, dimensionless quantities

.. math::

    \begin{align*}
        g_1(z) &= \left(
            3 - b_\mathrm{e} - \frac{3}{2} \Omega_\mathrm{m}
        \right) - (2 - 5s) \left(
            1 - \frac{1}{\mathcal{H} \chi}
        \right) \,, \\
        g_1(z) &= \left(
            3 - b_\mathrm{e} - \frac{3}{2} \Omega_\mathrm{m}
        \right) f - \frac{3}{2} \Omega_\mathrm{m} \big[
            g_1(z) - (2 - 5s)
        \big]
    \end{align*}

with evolution bias :math:`b_\mathrm{e}(z)` and magnification bias
:math:`s(z)`, where :math:`\mathcal{H}` is the conformal Hubble parameter,
:math:`\chi(z)` is the comoving distance, and the matter density
parameter evolves as

.. math::

    \Omega_\mathrm{m} = \frac{H_0^2}{H(z)^2}
        \Omega_{\mathrm{m},0} (1 + z)^3 \,.

Modifications to power spectrum multipoles from the relativistic
corrections are

.. math::

    \begin{align*}
        \Delta P_0(k, z) &= \left[
            \left(
                2 b_1 g_2 + \frac{2}{3} f g_2 + \frac{1}{3} f^2 g_1^2
            \right) \frac{\mathcal{H}^2}{k^2}
            + g_2^2 \frac{\mathcal{H}^4}{k^4}
        \right] P_\mathrm{m}(k,z) \,,\\
        \Delta P_2(k, z) &= \frac{2}{3} \left( 2 f g_2 + f^2 g_1^2 \right)
            \frac{\mathcal{H}^2}{k^2} P_\mathrm{m}(k,z) \,.
    \end{align*}

.. autosummary::

    relativistic_correction_func
    relativistic_correction_value
    relativistic_correction_factor

|

"""
# pylint: disable=no-name-in-module
from astropy.constants import c
from nbodykit.lab import cosmology as nbodykit_cosmology

_SPEED_OF_LIGHT_IN_KM_PER_S = c.to('km/s').value
_SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY = 1.686
_GROWTH_FACTOR_NORMALISATION = 1.27

FIDUCIAL_COSMOLOGY = nbodykit_cosmology.Planck15
r""":class:`nbodykit.cosmology.Cosmology`: Default Planck15 cosmology.

"""


def standard_kaiser_factor(order, bias, redshift, cosmo=FIDUCIAL_COSMOLOGY):
    r"""Compute the standard Kaiser power spectrum multipoles as multiples
    of the matter power spectrum, i.e. :math:`P_\ell/P_\mathrm{m}`.

    Parameters
    ----------
    order : int
        Order of the multipole, ``order >= 0``.
    bias : callable or float
        Scale-independent linear bias (at `redshift`).
    redshift : float
        Redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Cosmological model (default is ``FIDUCIAL_COSMOLOGY``).

    Returns
    -------
    factor : float
        Power spectrum multipole as a multiple of the matter power
        spectrum.

    """
    b_1 = bias(redshift) if callable(bias) else bias

    f = cosmo.scale_independent_growth_rate(redshift)

    if order == 0:
        factor = b_1 ** 2 + 2./3. * f * b_1 + 1./5. * f**2
    elif order == 2:
        factor = 4./3. * f * b_1 + 4./7. * f**2
    elif order == 4:
        factor = 8./35. * f**2
    else:
        factor = 0.

    return factor


def scale_dependence_kernel(redshift, cosmo=FIDUCIAL_COSMOLOGY):
    r"""Return the scale-dependence kernel :math:`A(k, z)` in the presence
    of local primordial non-Gaussianity.

    Parameters
    ----------
    redshift : float
        Redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Cosmological model (default is ``FIDUCIAL_COSMOLOGY``).

    Returns
    -------
    callable
        Scale-dependence kernel as a function of wavenumber (in
        :math:`h/\mathrm{Mpc}`).

    """
    numerical_constants = 3 * _SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY \
        * cosmo.Om0 * _GROWTH_FACTOR_NORMALISATION \
        * (FIDUCIAL_COSMOLOGY.H0 / _SPEED_OF_LIGHT_IN_KM_PER_S)**2

    transfer_function = \
        nbodykit_cosmology.power.transfers.CLASS(cosmo, redshift)

    return lambda k: numerical_constants / transfer_function(k)


def non_gaussianity_correction_factor(wavenumber, order, local_png, bias,
                                      redshift, cosmo=FIDUCIAL_COSMOLOGY,
                                      tracer_p=1.):
    r"""Compute modifications to the power spectrum multipoles by local
    primordial non-Gaussianity as multiples of the matter power spectrum,
    i.e. :math:`\Delta P_\ell/P_\mathrm{m}`.

    Parameters
    ----------
    wavenumber : float, array_like
        Wavenumber (in :math:`h/\mathrm{Mpc}`).
    order : int
        Order of the multipole, ``order >= 0``.
    local_png : float
        Local primordial non-Gaussianity.
    bias : callable or float
        Scale-independent linear bias (at `redshift`).
    redshift : float
        Redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    tracer_p : float, optional
        Tracer parameter (default is 1.).

    Returns
    -------
    factor : float :class:`numpy.ndarray`
        Power spectrum multipole modifications as multiples of the matter
        power spectrum.

    """
    f_nl, p = local_png, tracer_p
    b_1 = bias(redshift) if callable(bias) else bias

    f = cosmo.scale_independent_growth_rate(redshift)

    delta_b = f_nl * (b_1 - p) \
        * scale_dependence_kernel(redshift, cosmo=cosmo)(wavenumber) \
        / wavenumber ** 2

    if order == 0:
        factor = (2 * b_1 + 2./3. * f) * delta_b + delta_b ** 2
    elif order == 2:
        factor = 4./3. * f * delta_b
    else:
        factor = 0.

    return factor


def relativistic_correction_func(corr_order, cosmo=FIDUCIAL_COSMOLOGY,
                                 evolution_bias=None, magnification_bias=None):
    r"""Return the relativistic correction function
    :math:`g_1(z)` or :math:`g_2(z)`.

    Parameters
    ----------
    corr_order : {1, 2}, int
        Order of the correction parameter :math:`:\mathcal{H}/k`.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    evolution_bias, magnification_bias : callable or None, optional
        Evolution bias or magnification bias as a function of redshift
        (default is `None`).

    Returns
    -------
    callable
        Relativistic correction function as a function of redshift
        (in :math:`h`/Mpc).

    """
    astropy_cosmo = cosmo.to_astropy()

    b_e = evolution_bias or (lambda z: 0.)
    s = magnification_bias or (lambda z: 0.)
    omega_m = lambda z: astropy_cosmo.Om0 \
        * (astropy_cosmo.H0 / astropy_cosmo.H(z)) ** 2 * (1 + z) ** 3
    aH_chi = lambda z: astropy_cosmo.scale_factor(z) \
        * (astropy_cosmo.H(z).value / _SPEED_OF_LIGHT_IN_KM_PER_S) \
        * astropy_cosmo.comoving_distance(z).value
    f = cosmo.scale_independent_growth_rate

    def g1_of_z(z):
        return 3 - b_e(z) - 3./2. * omega_m(z) \
            - (2 - 5 * s(z)) * (1 - 1 / aH_chi(z))

    def g2_of_z(z):
        return (3 - b_e(z) - 3./2. * omega_m(z)) * f(z) \
            - 3./2. * omega_m(z) * (g1_of_z(z) - (2 - 5 * s(z)))

    if corr_order == 1:
        return g1_of_z
    if corr_order == 2:
        return g2_of_z
    raise ValueError("Accepted correction order is 1 or 2.")


def relativistic_correction_value(redshift, corr_order,
                                  cosmo=FIDUCIAL_COSMOLOGY,
                                  evolution_bias=None,
                                  magnification_bias=None):
    r"""Evaluate the redshift-dependent relativistic correction value
    :math:`g_1(z)` or :math:`g_2(z)`.

    Parameters
    ----------
    redshift : float
        Redshift.
    corr_order : {1, 2}, int
        Correction order of the parameter :math:`:\mathcal{H}/k`.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    evolution_bias, magnification_bias : float or None, optional
        Evolution bias or magnification bias evaluated at input `redshift`
        (default is `None`).

    Returns
    -------
    correction_value : float
        Relativistic correction function value (in :math:`h`/Mpc)
        at `redshift`.

    """
    astropy_cosmo = cosmo.to_astropy()

    b_e = evolution_bias or 0.
    s = magnification_bias or 0.
    omega_m = astropy_cosmo.Om0 \
        * (astropy_cosmo.H0 / astropy_cosmo.H(redshift)) ** 2 \
        * (1 + redshift) ** 3
    aH_chi = astropy_cosmo.scale_factor(redshift) \
        * (astropy_cosmo.H(redshift).value / _SPEED_OF_LIGHT_IN_KM_PER_S) \
        * astropy_cosmo.comoving_distance(redshift).value
    f = cosmo.scale_independent_growth_rate(redshift)

    g_1 = 3 - b_e - 3./2. * omega_m - (2 - 5 * s) * (1 - 1 / aH_chi)

    if corr_order == 1:
        return g_1
    if corr_order == 2:
        return (3 - b_e - 3./2. * omega_m) * f \
            - 3./2. * omega_m * (g_1 - (2 - 5 * s))
    raise ValueError("Accepted correction order is 1 or 2.")


def relativistic_correction_factor(wavenumber, order, redshift, bias,
                                   correction_value_1=None,
                                   correction_value_2=None,
                                   cosmo=FIDUCIAL_COSMOLOGY,
                                   evolution_bias=None,
                                   magnification_bias=None):
    r"""Compute modifications to the power spectrum multipoles by
    relativistic corrections as multiples of the matter power spectrum,
    i.e. :math:`\Delta P_\ell/P_\mathrm{m}`.

    Parameters
    ----------
    wavenumber : float, array_like
        Wavenumber (in :math:`h/\mathrm{Mpc}`).
    order : int
        Order of the multipole, ``order >= 0``.
    redshift : float
        Redshift.
    bias : callable or float
        Scale-independent linear bias (at `redshift`).
    correction_value_1, correction_value_2 : float or None, optional
        If not `None` (default), this is directly used as
        :math:`g_1(z)` or :math:`g_2(z)` at `redshift` in calculations,
        and `evolution_bias`, `magnification_bias` are ignored.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    evolution_bias : float or callable or None, optional
        Evolution bias as a function of redshift, or evaluated at input
        `redshift` (default is `None`).  If callable/float,
        `magnification_bias` must also be callable/float.
    magnification_bias : float or callable or None, optional
        Magnification bias as a function of redshift, or evaluated at input
        `redshift` (default is `None`).  If callable/float,
        `evolution_bias` must also be callable/float.

    Returns
    -------
    factor : float :class:`numpy.ndarray`
        Power spectrum multipole modifications as multiples of the matter
        power spectrum.

    """
    astropy_cosmo = cosmo.to_astropy()

    aH_over_k = astropy_cosmo.scale_factor(redshift) \
        * (astropy_cosmo.H(redshift).value / _SPEED_OF_LIGHT_IN_KM_PER_S) \
        / wavenumber

    b_1 = bias(redshift) if callable(bias) else bias
    f = cosmo.scale_independent_growth_rate(redshift)

    g_1 = correction_value_1 or relativistic_correction_func(
        1, cosmo=cosmo,
        evolution_bias=evolution_bias,
        magnification_bias=magnification_bias
    )(redshift)
    g_2 = correction_value_2 or relativistic_correction_func(
        2, cosmo=cosmo,
        evolution_bias=evolution_bias,
        magnification_bias=magnification_bias
    )(redshift)

    if order == 0:
        factor = (2 * b_1 * g_2 + 2./3. * f * g_2 + 1./3. * f**2 * g_1 ** 2) \
            * aH_over_k ** 2 + g_2**2 * aH_over_k ** 4
    elif order == 2:
        factor = 2./3. * (2 * f * g_2 + f**2 * g_1 ** 2) * aH_over_k ** 2
    else:
        factor = 0.

    return factor
