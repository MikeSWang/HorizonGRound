"""
Modification (:mod:`modification`)
===========================================================================

Scale-dependent modifications to the clustering power spectrum from
relativistic corrections to the over-density field :math:`\delta` or from
scale-dependent linear bias induced by local primordial non-Gaussianity
:math:`f_\textrm{NL}`.

Relativistic corrections
------------------------

The redshift-dependent correction function takes the form

.. math::

    g(z) = \frac{\dot{\mathcal{H}}}{\mathcal{H}} + \mathcal{H} \left(
        \frac{2 - 5s}{\mathcal{H} \chi} + 5s - f_\mathrm{ev}
    \right)

with evolution bias :math:`f_\mathrm{ev}(z)` and magnification bias
:math:`s(z)`, so that the Fourier-space 2-point correlator in the global
plane parallel approximation :math:`\mu \equiv \hat{\mathbf{k}}
\mathbf{\cdot} \hat{\mathbf{n}}` is modified by an amount of

.. math::

    \Delta \left\langle
        \left\vert \delta(z, \mathbf{k}) \right\vert^2
    \right\rangle = g(z)^2 f(z)^2 (\mu^2/k^2) P_\mathrm{m}(k, z) \,,

where :math:`f(z)` is the linear growth rate and :math:`P_\mathrm{m}` is
the matter power spectrum.

We can rearrange this as the sum of three terms

.. math::

    g(z) = \underbrace{
        \frac{2}{\chi} + \mathcal{H} \left[
            1 - \frac{3}{2} \Omega_\mathrm{m,0} (1 + z)^3
        \right]
    }_{\textrm{geometric}}
    \phantom{+} \underbrace{
        - \mathcal{H} f_\mathrm{ev}(z)}_{\textrm{evolution}
    } + \underbrace{
        5s(z) \left( \mathcal{H} - \frac{1}{\chi} \right)
    }_{\text{lensing}} \,.


"""
import numpy as np
from nbodykit.lab import cosmology

FIDUCIAL_COSMOLOGY = cosmology.Planck15
r""":class:`nbodykit.cosmology.cosmology.Cosmology`: Default *Planck*15
cosmology.

"""


def scale_dependence_kernel(redshift, cosmo=FIDUCIAL_COSMOLOGY):
    """Return the scale-dependence kernel in the presence of local
    primordial non-Gaussianity as a function of redshift.

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
    SPEED_OF_LIGHT_IN_KM_PER_S = 2998792.

    numerical_constants = 3 * (100*cosmo.h / SPEED_OF_LIGHT_IN_KM_PER_S)**2 \
        * SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY * cosmo.Omega0_m
    transfer_function = cosmology.power.transfers.CLASS(cosmo, redshift)

    return lambda k: numerical_constants / transfer_function(k)


def relativistic_corrections(cosmo=FIDUCIAL_COSMOLOGY, geometric_bias=True,
                             evolution_bias=None, magnification_bias=None):
    """Return the general relativistic corrections as a function of
    redshift.

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
    background =  cosmology.background.MatterDominated(cosmo.Omega0_m)

    a = lambda z: (1 + z)**(-1)
    H_conformal = lambda z: 100 * cosmo.h * a(z) * background.E(a(z))
    chi = lambda z: cosmo.comoving_distance(z)

    if geometric_bias:
        geometric_term = lambda z: 0
    else:
        geometric_term = lambda z: \
            2 / chi(z) + H_conformal(z) * (1 - 3/2*cosmo.Omega0_m / a(z)**3)

    if evolution_bias is None:
        evolution_term = lambda z: 0.
    else:
        evolution_term = lambda z: \
            - H_conformal(z) * evolution_bias(z)

    if magnification_bias is None:
        lensing_term = lambda z: 0.
    else:
        lensing_term = lambda z: \
            5 * magnification_bias(z) * (H_conformal(z) - 1 / chi(z))

    return np.vectorize(
        lambda z: geometric_term(z) + evolution_term(z) + lensing_term(z)
    )


def relativistic_modification(wavenumber, redshift, multipole,
                              cosmo=FIDUCIAL_COSMOLOGY, geometric_bias=True,
                              evolution_bias=None, magnification_bias=None):
    """Power spectrum multipole modification by general relativistic
    corrections as multiples of the matter power spectrum.

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
        * cosmo.scale_independent_growth_factor(redshift)**2 \
        / wavenumber**2

    if multipole == 0:
        modification_factor = 1/3 * modification
    elif multipole == 2:
        modification_factor = 2/3 * modification
    else:
        modification_factor = np.zeros(len(np.atleast_1d(wavenumber)))

    return modification_factor


def non_gaussianity_modification(wavenumber, redshift, multipole, f_nl, b_1,
                                 cosmo=FIDUCIAL_COSMOLOGY):
    """Power spectrum multipole modification by local primordial
    non-Gaussianity as multiples of the matter power spectrum.

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

    Returns
    -------
    modification_factor : float :class:`numpy.ndarray`
        Power spectrum multipole modification as multiples of the
        matter power spectrum.

    """
    f = cosmo.scale_independent_growth_factor(redshift)

    modification = f_nl * (b_1 - 1) \
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
    """Kaiser RSD model power spectrum multipoles as multiples of the
    matter power spectrum.

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
    f = cosmo.scale_independent_growth_factor(redshift)

    if multipole == 0:
        factor = 1 + 2/3 * f * b_1 + 1/5 * f**2
    elif multipole == 2:
        factor = 4/3 * f * b_1 + 4/7 * f**2
    elif multipole == 4:
        factor = 8/35 * f**2
    else:
        factor = 0.

    return factor
