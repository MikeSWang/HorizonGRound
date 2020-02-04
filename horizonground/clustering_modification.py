r"""
Clustering modification (:mod:`~horizonground.clustering_modification`)
===========================================================================

Scale-dependent modifications to Newtonian clustering statistics from
relativistic corrections or from local primordial non-Gaussianity.

.. autosummary::

    standard_kaiser_modification


Non-Gaussianity modifications
---------------------------------------------------------------------------

Local primordial non-Gaussianty :math:`f_\textrm{NL}` induces scale
dependence

.. math::

    A(k, z) = 3 \left( \frac{H_0}{\mathrm{c}} \right)^2
        \frac{1.27 \Omega_\mathrm{m,0} \delta_\mathrm{c}}{D(z)T(k)}

where the growth factor :math:`D(z)` is normalised to unity at the current
epoch (thus the numerical factor 1.27), and the transfer function
:math:`T(k)` is normalised to unity at the vanishing wavenumber, so that
linear bias :math:`b_1(z)` is modified by

.. math::

    \Delta b(k, z) = f_\mathrm{NL} [b_1(z) - p] \frac{A(k, z)}{k^2} \,,

where :math:`p` is a tracer species--dependent parameter.

.. autosummary::

    scale_dependence_kernel
    non_gaussianity_modification


Relativistic corrections
---------------------------------------------------------------------------

The redshift-dependent correction function takes the form

.. math::

    g(z) = \mathcal{H} \left(
        \frac{\mathcal{H}'}{\mathcal{H}^2}
        + \frac{2 - 5s}{\mathcal{H} \chi} + 5s - f_\mathrm{ev}
    \right)

with evolution bias :math:`f_\mathrm{ev}(z)` and magnification bias
:math:`s(z)`, so that the Fourier-space 2-point correlator in the global
plane parallel approximation :math:`\mu \equiv \hat{\mathbf{k}}
\mathbf{\cdot} \hat{\mathbf{n}}` is modified by an amount of

.. math::

    \Delta \left\langle
        \left\vert \delta(z, \mathbf{k}) \right\vert^2
    \right\rangle = g(z)^2 f(z)^2 \frac{\mu^2}{k^2} P_\mathrm{m}(k, z) \,,

where :math:`f(z)` is the linear growth rate and :math:`P_\mathrm{m}` is
the matter power spectrum.

We can rearrange this as the sum of three terms

.. math::

    g(z)/\mathcal{H}(z) = \underbrace{
        \frac{2}{\mathcal{H}\chi} + \left[
            1 - \frac{3}{2} \Omega_\mathrm{m,0} (1 + z)^3
        \right]
    }_{\textrm{geometric}}
    + \underbrace{
        f_\mathrm{ev}(z)}_{\textrm{evolution}
    } + \underbrace{
        5s(z) \left( 1 - \frac{1}{\mathcal{H}\chi} \right)
    }_{\text{magnification}} \,.

.. autosummary::

    relativistic_corrections
    relativistic_modification

|

"""
from __future__ import division

import numpy as np
from nbodykit.lab import cosmology as nbk_cosmology

FIDUCIAL_COSMOLOGY = nbk_cosmology.Planck15
r""":class:`nbodykit.cosmology.Cosmology`: Default Planck15 cosmology.

"""

_SPEED_OF_LIGHT_IN_KM_PER_S = 2998792.


def scale_dependence_kernel(redshift, cosmo=FIDUCIAL_COSMOLOGY):
    r"""Return the scale-dependence kernel :math:`A(k;z)` in the presence
    of local primordial non-Gaussianity as a function of wavenumber at a
    given redshift.

    Parameters
    ----------
    redshift : float
        Redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).

    Returns
    -------
    callable
        Scale-dependence kernel as a function of wavenumber (in h/Mpc).

    """
    SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY = 1.686
    EQUALITY_NORMALISATION = 1.27

    numerical_constants = 3 * SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY \
        * cosmo.Om0 * EQUALITY_NORMALISATION \
        * (100*FIDUCIAL_COSMOLOGY.h / _SPEED_OF_LIGHT_IN_KM_PER_S)**2 \

    transfer_function = nbk_cosmology.power.transfers.CLASS(cosmo, redshift)

    return lambda k: numerical_constants / transfer_function(k)


def relativistic_corrections(cosmo=FIDUCIAL_COSMOLOGY, geometric_bias=True,
                             evolution_bias=None, magnification_bias=None):
    r"""Return the general relativistic corrections :math:`g(z)` as a
    function of redshift.

    Parameters
    ----------
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    geometric_bias : bool, optional
        If `True` (default), include geometric bias.
    evolution_bias, magnification_bias : callable or None, optional
        Evolution bias or magnification bias as a function of redshift
        (default is `None`).

    Returns
    -------
    callable
        Relativistic corrections as a function of redshift.

    """
    astropy_cosmo = cosmo.to_astropy()

    a = astropy_cosmo.scale_factor
    aH = lambda z: a(z) * astropy_cosmo.H(z).value / _SPEED_OF_LIGHT_IN_KM_PER_S
    chi = lambda z: astropy_cosmo.comoving_distance(z).value

    if geometric_bias:
        geometric_term = lambda z: \
            2 / chi(z) + aH(z) * (1 - 3/2*astropy_cosmo.Om0 / a(z)**3)
    else:
        geometric_term = lambda z: 0

    if evolution_bias is None:
        evolution_term = lambda z: 0.
    else:
        evolution_term = lambda z: - aH(z) * evolution_bias(z)

    if magnification_bias is None:
        lensing_term = lambda z: 0.
    else:
        lensing_term = lambda z: \
            5 * magnification_bias(z) * (aH(z) - 1/chi(z))

    return np.vectorize(
        lambda z: (geometric_term(z) + evolution_term(z) + lensing_term(z))
    )


def relativistic_modification(wavenumber, redshift, multipole,
                              cosmo=FIDUCIAL_COSMOLOGY, geometric_bias=True,
                              evolution_bias=None, magnification_bias=None):
    r"""Power spectrum multipole modification by general relativistic
    corrections as multiples of the matter power spectrum

    .. math::

        \begin{align*}
            \frac{\Delta P_0(k; z)}{P_\textrm{m}(k,z)} &= \frac{1}{3}
                \frac{g(z)^2 f(z)^2}{k^2} \,,\\
            \frac{\Delta P_2(k; z)}{P_\textrm{m}(k,z)} &= \frac{2}{3}
                \frac{g(z)^2 f(z)^2}{k^2} \,,
        \end{align*}

    where :math:`f(z)` is the linear growth rate.

    Parameters
    ----------
    wavenumber : float, array_like
        Wavenumber (in h/Mpc).
    redshift : float
        Redshift.
    multipole : int
        Order of the multipole, ``multipole >= 0``.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    geometric_bias : bool, optional
        If `True` (default), include geometric bias.
    evolution_bias, magnification_bias : callable or None, optional
        Evolution bias or magnification bias as a function of redshift
        (default is `None`).

    Returns
    -------
    modification_factor : float :class:`numpy.ndarray`
        Power spectrum multipole modification as multiples of the
        matter power spectrum.

    """
    correction_function = relativistic_corrections(
        cosmo=cosmo,
        geometric_bias=geometric_bias,
        evolution_bias=evolution_bias,
        magnification_bias=magnification_bias
    )

    modification = correction_function(redshift)**2 \
        * cosmo.scale_independent_growth_rate(redshift)**2 \
        / wavenumber**2

    if multipole == 0:
        modification_factor = 1/3 * modification
    elif multipole == 2:
        modification_factor = 2/3 * modification
    else:
        modification_factor = np.zeros(len(np.atleast_1d(wavenumber)))

    return modification_factor


def non_gaussianity_modification(wavenumber, redshift, multipole, f_nl, b_1,
                                 cosmo=FIDUCIAL_COSMOLOGY, tracer_p=1.):
    r"""Power spectrum multipole modification by local primordial
    non-Gaussianity as multiples of the matter power spectrum

    .. math::

        \begin{align*}
            \frac{\Delta P_0(k, z)}{P_\textrm{m}(k,z)} &= \left(
                2 b_1 + \frac{2}{3} f
            \right) f_\textrm{NL} (b_1 - 1) \frac{A(k)}{k^2}
            + \left[
                f f_\textrm{NL} (b_1 - 1) \frac{A(k)}{k^2}
            \right]^2 \,, \\
            \frac{\Delta P_2(k, z)}{P_\textrm{m}(k,z)} &= \frac{4}{3} f
                f_\textrm{NL} (b_1 - 1) \frac{A(k)}{k^2} \,.
        \end{align*}

    Parameters
    ----------
    wavenumber : float, array_like
        Wavenumber (in h/Mpc).
    redshift : float
        Redshift.
    multipole : int
        Order of the multipole, ``multipole >= 0``.
    f_nl : float
        Local primordial non-Gaussianity.
    b_1 : float
        Scale-independent linear bias at `redshift`.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    tracer_p : float, optional
        Tracer parameter (default is 1.).

    Returns
    -------
    modification_factor : float :class:`numpy.ndarray`
        Power spectrum multipole modification as multiples of the
        matter power spectrum.

    """
    f = cosmo.scale_independent_growth_rate(redshift)

    modification = f_nl * (b_1 - tracer_p) \
        * scale_dependence_kernel(redshift, cosmo=cosmo)(wavenumber) \
        / wavenumber**2

    if multipole == 0:
        modification_factor = (2*b_1 + 2/3*f) * modification + modification**2
    elif multipole == 2:
        modification_factor = 4/3 * f * modification
    else:
        modification_factor = np.zeros(len(np.atleast_1d(wavenumber)))

    return modification_factor


def standard_kaiser_modification(redshift, multipole, b_1,
                                 cosmo=FIDUCIAL_COSMOLOGY):
    r"""Kaiser RSD model power spectrum multipoles as multiples of the
    matter power spectrum

    .. math::

        \begin{align*}
            \frac{P_0(k; z)}{P_\textrm{m}(k;z)} &= b_1(z)^2
                + \frac{2}{3} b_1(z) f(z) + \frac{1}{5} f(z)^2 \,, \\
            \frac{P_2(k; z)}{P_\textrm{m}(k;z)} &=
                \frac{4}{3} b_1(z) f(z) + \frac{4}{7} f(z)^2 \,, \\
            \frac{P_4(k; z)}{P_\textrm{m}(k;z)} &= \frac{8}{35}
                f(z)^2 \,.
        \end{align*}

    Parameters
    ----------
    redshift : float
        Redshift.
    multipole : int
        Order of the multipole, ``multipole >= 0``.
    b_1 : float
        Scale-independent linear bias at the same redshift as `redshift`.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).

    Returns
    -------
    factor : float
        Power spectrum multipoled as multiples of the matter power
        spectrum.

    """
    f = cosmo.scale_independent_growth_rate(redshift)

    if multipole == 0:
        factor = b_1 ** 2 + 2/3 * f * b_1 + 1/5 * f**2
    elif multipole == 2:
        factor = 4/3 * f * b_1 + 4/7 * f**2
    elif multipole == 4:
        factor = 8/35 * f**2
    else:
        factor = 0.

    return factor
