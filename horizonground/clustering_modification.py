r"""
Clustering modification (:mod:`~horizonground.clustering_modification`)
===========================================================================

Modifications to the isotropic tracer clustering statistics (power
spectrum) in Newtonian prescription in the distant-observer limit and
plane-parallel approximation.


Standard Kaiser RSD model
---------------------------------------------------------------------------

Redshift-space distortions induces anisotropy in the power spectrum
multipoles

.. math::

    \begin{align*}
        \Delta P_0(k, z) &=
            \left[
                \frac{2}{3} b_1(z) f(z) + \frac{1}{5} f(z)^2
            \right] P_\mathrm{m}(k, z) \,, \\
        \Delta P_2(k, z) &=
            \left[
                \frac{4}{3} b_1(z) f(z) + \frac{4}{7} f(z)^2
            \right] P_\mathrm{m}(k, z) \,, \\
        \Delta P_4(k, z) &= \frac{8}{35} f(z)^2 P_\mathrm{m}(k, z) \,
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
        \Delta b_k(z) = f_\mathrm{NL} [b_1(z) - p] \frac{A(k, z)}{k^2} \,,

where the scale-dependence kernel is

.. math::

    A(k, z) = 3 \left( \frac{H_0}{\mathrm{c}} \right)^2
        \frac{1.27 \Omega_\mathrm{m,0} \delta_\mathrm{c}}{D(z)T(k)} \,.

Here :math:`H_0` is the Hubble parameter at the current epoch
(in km/s/Mpc), :math:`\mathrm{c}` the speed of light,
:math:`\Omega_\mathrm{m,0}` the matter density parameter at the current
epoch, and :math:`\delta_\mathrm{c}` the critical over-density in
gravitational spherical collapse.  The growth factor :math:`D(z)` is
normalised to unity at the current epoch (thus the numerical factor 1.27),
the transfer function :math:`T(k)` is normalised to unity as
:math:`k \to 0`, and :math:`p` is a tracer-dependent parameter.

Modifications to power spectrum multipoles as a result of local primordial
non-Gaussianty are

.. math::

    \begin{align*}
        \Delta P_0(k, z) &= \left[
            2 b_1 \Delta b(k, z)
            + \frac{2}{3} f \Delta b(k, z)
            + f^2 \Delta b(k, z)^2
        \right] P_\mathrm{m}(k, z) \,, \\
        \Delta P_2(k, z) &=
            \frac{4}{3} f \Delta b(k, z) P_\mathrm{m}(k, z) \,.
    \end{align*}

.. autosummary::

    scale_dependence_kernel
    non_gaussianity_factor


Relativistic corrections
---------------------------------------------------------------------------

The relativistic corrections to the Newtonian tracer clustering mode

.. math::

    \delta(\mathbf{k}, z) \mapsto \delta(\mathbf{k}, z)
            + \frac{g(z)}{\mathcal{H}(z)} v_{\parallel}(\mathbf{k}, z)
        = \delta(\mathbf{k}, z)
            + \mathrm{i} g(z) \frac{f(z)}{b_1(z)} \frac{\mu}{k}
                \delta(\mathbf{k}, z)

have the redshift dependence

.. math::

    \frac{g(z)}{\mathcal{H}(z)} =
        \frac{\mathcal{H}'}{\mathcal{H}^2}
        + 5s + \frac{2 - 5s}{\mathcal{H} \chi}
        - f_\mathrm{ev}

with evolution bias :math:`f_\mathrm{ev}(z)` and magnification bias
:math:`s(z)`, where :math:`v_{\parallel}` is the line-of-sight velocity,
:math:`\chi(z)` is the comoving distance and :math:`'` denotes derivatives
with respect to the conformal time. This can be treated as the sum of three
terms

.. math::

    \frac{g(z)}{\mathcal{H}(z)} = \underbrace{
        \frac{2}{\mathcal{H}\chi} + \left[
            1 - \frac{3}{2} \Omega_\mathrm{m,0} (1 + z)^3
        \right]
    }_{\text{geometric}}
    \: \underbrace{- f_\mathrm{ev}(z)}_{\text{evolution}}
    + \underbrace{
        5s(z) \left( 1 - \frac{1}{\mathcal{H}\chi} \right)
    }_{\text{magnification}} \,.

Modifications to power spectrum multipoles as a result of the relativistic
corrections are

.. math::

    \begin{align*}
        \Delta P_0(k, z) &= \frac{1}{3}
            \frac{g(z)^2 f(z)^2}{k^2} P_\mathrm{m}(k,z) \,,\\
        \Delta P_2(k, z) &= \frac{2}{3}
            \frac{g(z)^2 f(z)^2}{k^2} P_\mathrm{m}(k,z) \,,
    \end{align*}

.. autosummary::

    relativistic_correction_func
    relativistic_factor

|

"""
from __future__ import division

import numpy as np
from nbodykit.lab import cosmology as nbk_cosmology

_SPEED_OF_LIGHT_IN_KM_PER_S = 2998792.

FIDUCIAL_COSMOLOGY = nbk_cosmology.Planck15
r""":class:`nbodykit.cosmology.Cosmology`: Default Planck15 cosmology.

"""


def standard_kaiser_factor(order, bias, redshift, cosmo=FIDUCIAL_COSMOLOGY):
    r"""Compute the standard Kaiser power spectrum multipoles as multiples
    of the matter power spectrum.

    Parameters
    ----------
    order : int
        Order of the multipole, ``order >= 0``.
    bias : float
        Scale-independent linear bias at `redshift`.
    redshift : float
        Redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).

    Returns
    -------
    factor : float
        Power spectrum multipoles as multiples of the matter power
        spectrum.

    """
    b_1 = bias

    f = cosmo.scale_independent_growth_rate(redshift)

    if order == 0:
        factor = b_1 ** 2 + 2/3 * f * b_1 + 1/5 * f**2
    elif order == 2:
        factor = 4/3 * f * b_1 + 4/7 * f**2
    elif order == 4:
        factor = 8/35 * f**2
    else:
        factor = 0.

    return factor


def scale_dependence_kernel(redshift, cosmo=FIDUCIAL_COSMOLOGY):
    r"""Return the scale-dependence kernel :math:`A(k, z)` in the presence
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
        Scale-dependence kernel as a function of wavenumber (in
        :math:`h/\textrm{Mpc}`).

    """
    SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY = 1.686
    EQUALITY_NORMALISATION = 1.27

    numerical_constants = 3 * SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY \
        * cosmo.Om0 * EQUALITY_NORMALISATION \
        * (100*FIDUCIAL_COSMOLOGY.h / _SPEED_OF_LIGHT_IN_KM_PER_S)**2 \

    transfer_function = nbk_cosmology.power.transfers.CLASS(cosmo, redshift)

    return lambda k: numerical_constants / transfer_function(k)


def non_gaussianity_factor(wavenumber, order, local_NG, bias, redshift,
                           cosmo=FIDUCIAL_COSMOLOGY, tracer_param=1.):
    r"""Compute the power spectrum multipoles modified by local primordial
    non-Gaussianity as multiples of the matter power spectrum.

    Parameters
    ----------
    wavenumber : float, array_like
        Wavenumber (in :math:`h/\textrm{Mpc}`).
    order : int
        Order of the multipole, ``order >= 0``.
    local_NG : float
        Local primordial non-Gaussianity.
    bias : float
        Scale-independent linear bias at `redshift`.
    redshift : float
        Redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    tracer_param : float, optional
        Tracer parameter (default is 1.).

    Returns
    -------
    factor : float :class:`numpy.ndarray`
        Power spectrum multipoles as multiples of the matter power
        spectrum.

    """
    f_nl, b_1, p = local_NG, bias, tracer_param

    f = cosmo.scale_independent_growth_rate(redshift)

    delta_b = f_nl * (b_1 - p) \
        * scale_dependence_kernel(redshift, cosmo=cosmo)(wavenumber) \
        / wavenumber**2

    if order == 0:
        factor = b_1**2 + (2*b_1 + 2/3*f) * delta_b + delta_b**2
    elif order == 2:
        factor = 4/3 * f * delta_b
    else:
        factor = 0.

    return factor


def relativistic_correction_func(cosmo=FIDUCIAL_COSMOLOGY, geometric=True,
                                 evolution_bias=None, magnification_bias=None):
    r"""Return the relativistic correction redshift-dependence function
    :math:`g(z)` as a function of redshift.

    Parameters
    ----------
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    geometric : bool, optional
        If `True` (default), include geometric perturbations.
    evolution_bias, magnification_bias : callable or None, optional
        Evolution bias or magnification bias as a function of redshift
        (default is `None`).

    Returns
    -------
    callable
        Relativistic correction redshift-dependence function as a function
        of redshift.

    """
    astropy_cosmo = cosmo.to_astropy()

    a = astropy_cosmo.scale_factor
    aH = lambda z: a(z) * astropy_cosmo.H(z).value / _SPEED_OF_LIGHT_IN_KM_PER_S
    chi = lambda z: astropy_cosmo.comoving_distance(z).value

    if geometric:
        geometric_term = lambda z: \
            2 / chi(z) + aH(z) * (1 - 3/2 * astropy_cosmo.Om0 / a(z)**3)
    else:
        geometric_term = lambda z: 0.

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


def relativistic_factor(wavenumber, order, redshift, cosmo=FIDUCIAL_COSMOLOGY,
                        geometric=True, evolution_bias=None,
                        magnification_bias=None):
    r"""Compute power spectrum multipole modified by relativistic
    corrections as multiples of the matter power spectrum.

    Parameters
    ----------
    wavenumber : float, array_like
        Wavenumber (in :math:`h/\textrm{Mpc}`).
    order : int
        Order of the multipole, ``order >= 0``.
    redshift : float
        Redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`, optional
        Base cosmological model (default is ``FIDUCIAL_COSMOLOGY``).
    geometric : bool, optional
        If `True` (default), include geometric perturbations.
    evolution_bias, magnification_bias : callable or None, optional
        Evolution bias or magnification bias as a function of redshift
        (default is `None`).

    Returns
    -------
    factor : float :class:`numpy.ndarray`
        Power spectrum multipoles as multiples of the matter power
        spectrum.

    """
    correction_function = relativistic_correction_func(
        cosmo=cosmo,
        geometric=geometric,
        evolution_bias=evolution_bias,
        magnification_bias=magnification_bias
    )

    modification = correction_function(redshift)**2 \
        * cosmo.scale_independent_growth_rate(redshift)**2 \
        / wavenumber**2

    if order == 0:
        factor = 1/3 * modification
    elif order == 2:
        factor = 2/3 * modification
    else:
        factor = 0.

    return factor
