"""Plot 2-d parameter constraints from parameter chains.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import cumtrapz, simps
from scipy.ndimage import gaussian_filter

try:
    from config.program import stylesheet
except ImportError:
    # pylint: disable=multiple-imports
    import os, sys
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, "".join([current_file_dir, "/../"]))
    from config.program import stylesheet

plt.style.use(stylesheet)

AREA_FILL_ALPHA = 1./3.
ONE_SIGMA_QUANTILES = [0.158655, 0.841345]
SIGMA_LEVELS = [0.864665, 0.393469, 0.000001]
legend_state = ([], [])


def gradient_colour_map(start_colour, end_colour, name):
    """Generate a colour map using gradients between two colours.

    Parameters
    ----------
    start_colour, end_colour : str
        Start or end colour as a HEX code string.
    name : str
        Name of the colour map.

    Returns
    -------
    cmap
        Generated colour map.

    """
    r0, g0, b0 = to_rgb(start_colour)
    r1, g1, b1 = to_rgb(end_colour)

    colour_dict = {
        'red': ((0, r0, r0), (1, r1, r1)),
        'green': ((0, g0, g0), (1, g1, g1)),
        'blue': ((0, b0, b0), (1, b1, b1))
    }

    cmap = LinearSegmentedColormap(name, colour_dict)

    return cmap


def convert_chains_to_grid(chains, bins=None, smooth=None,
                           range_x=None, range_y=None):
    """Convert 2-parameter chains to the join posterior binned over a
    2-d parameter grid.

    Parameters
    ----------
    chains : float, array_like
        Parameter chains.
    bins : int, array_like, optional
        Number of bins (for each paramter) (default is `None`).
    smooth : float or None, optional
        The standard deviation for Gaussian kernel passed to
        `scipy.ndimage.gaussian_filter` to smooth the posterior grid.
        If `None` (default), no smoothing is applied.
    range_x, range_y : float array_like, optional
        Parameter range (default is `None`).

    Returns
    -------
    posterior_grid, x_grid, y_grid : :class:`numpy.ndarray`
        Posterior and parameter grids from histogram binning.

    """
    x, y = np.transpose(chains)

    try:
        bin_range = [
            [min(range_x), max(range_x)], [min(range_y), max(range_y)]
        ]
    except TypeError:
        bin_range = None

    posterior_grid, x_edges, y_edges = np.histogram2d(
        x.flatten(), y.flatten(), bins=bins, range=bin_range
    )
    if smooth:
        posterior_grid = gaussian_filter(posterior_grid, smooth)

    x_grid = (x_edges[1:] + x_edges[:-1]) / 2
    y_grid = (y_edges[1:] + y_edges[:-1]) / 2

    return posterior_grid, x_grid, y_grid


def plot_2d_contours(posterior, x, y, x_range=None, y_range=None,
                     estimate=None, x_precision=None, y_precision=None,
                     x_label=None, y_label=None, cmap=None, alpha=None,
                     fig=None):
    """Plot 2-d contours from the joint parameter posterior on a grid.

    Parameters
    ----------
    posterior : float, array_like
        Posterior evaluations.
    x, y : float, array_like
        Parameter coordinates.
    x_range, y_range : sequence or None
        Parameter range as a sequence of length 2 (default is `None`).
    estimate : {'median', 'maximum', None}, optional
        Parameter estimate type, if any (default is `None`).
    x_precision, y_precision : int or None, optional
        Parameter precision as a number of decimal places (default is
        `None`).
    x_label, y_label : str or None
        Parameter label (default is `None`).
    cmap : str or None, optional
        Principal colour map (default is `None`).
    alpha : str or None, optional
        Principal alpha transparency (default is `None`).
    fig : :class:`matplotlib.figure.Figure` *or None*, optional
        Existing figure object to plot on.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Plotted figure object.
    x_estimate, y_estimate : tuple of float or None
        Parameter estimates with low and upper uncertainties.  `None`
        returned if `estimate` is `None`.
    patch : :class:`matplotlib.patches.Rectangle`
        A colour patch to be used in the legend.

    """
    # Set up the plottable grid.
    if x_range:
        x_selector = slice(
            np.argmin(np.abs(x - x_range[0])),
            np.argmin(np.abs(x - x_range[1])) + 1
        )
    else:
        x_selector = slice(None)

    if y_range:
        y_selector = slice(
            np.argmin(np.abs(y - y_range[0])),
            np.argmin(np.abs(y - y_range[1])) + 1
        )
    else:
        y_selector = slice(None)

    x, y = np.asarray(x)[x_selector], np.asarray(y)[y_selector]

    xx, yy = np.meshgrid(x, y, indexing='ij')

    posterior = np.asarray(posterior)[x_selector, y_selector]

    posterior /= simps([simps(xslice, y) for xslice in posterior], x)

    # Set up plottable areas.
    if fig is None:
        fig = plt.figure()
        xy_panel = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
        x_panel = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=xy_panel)
        y_panel = plt.subplot2grid((4, 4), (1, 3), rowspan=3, sharey=xy_panel)
    else:
        xy_panel, x_panel, y_panel = fig.axes

    # Locate posterior contours.
    h_flat = np.flip(np.sort(posterior.flatten()))
    cum_h = np.cumsum(h_flat)
    cum_h /= cum_h[-1]

    h_levels = np.zeros_like(SIGMA_LEVELS)
    for n_sigma, sigma_level in enumerate(SIGMA_LEVELS):
        try:
            h_levels[n_sigma] = h_flat[cum_h <= sigma_level][-1]
        except IndexError:
            h_levels[n_sigma] = h_flat[0]

    # Plot posterior contours.
    try:
        contour = xy_panel.contourf(
            xx, yy, posterior, h_levels, antialiased=True,
            cmap=cmap, alpha=alpha, zorder=2
        )
        primary_colour = contour.cmap(contour.cmap.N)
        patch = plt.Rectangle(
            (0, 0), 2./3., 1, ec=None,
            fc=contour.collections[-1].get_facecolor()[0]
        )
    except ValueError as error:
        if str(error) == "Contour levels must be increasing":
            raise ValueError(
                "Cannot process posterior values into contours."
            ) from error
        raise ValueError from error

    xy_panel.contour(
        contour, colors=primary_colour,
        alpha=min(2*alpha, 1.) if isinstance(alpha, float) else 1.,
        zorder=3
    )

    # Marginalise to PDFs.
    pdf_x = np.asarray([simps(xslice, y) for xslice in posterior])
    pdf_y = np.asarray([simps(yslice, x) for yslice in posterior.T])
    cdf_x = cumtrapz(pdf_x, x, initial=0.)
    cdf_y = cumtrapz(pdf_y, y, initial=0.)

    pdf_x /= cdf_x[-1]
    pdf_y /= cdf_y[-1]
    cdf_x /= cdf_x[-1]
    cdf_y /= cdf_y[-1]

    # Plot marginal posteriors.
    x_panel.plot(x, pdf_x, c=primary_colour, zorder=3)
    y_panel.plot(pdf_y, y, c=primary_colour, zorder=3)

    # Make estimates.
    if estimate:
        # Determine closest parameter estimate indices.
        if estimate == 'maximum':
            x_fit_idx, y_fit_idx = np.argmax(pdf_x), np.argmax(pdf_y)
        elif estimate == 'median':
            x_fit_idx = np.argmin(np.abs(cdf_x - 1./2.))
            y_fit_idx = np.argmin(np.abs(cdf_y - 1./2.))

        x_lower_idx = np.argmin(np.abs(cdf_x - ONE_SIGMA_QUANTILES[0]))
        y_lower_idx = np.argmin(np.abs(cdf_y - ONE_SIGMA_QUANTILES[0]))
        x_upper_idx = np.argmin(np.abs(cdf_x - ONE_SIGMA_QUANTILES[-1]))
        y_upper_idx = np.argmin(np.abs(cdf_y - ONE_SIGMA_QUANTILES[-1]))

        # Determined closest parameter estimates.
        x_fit, x_lower, x_upper = x[[x_fit_idx, x_lower_idx, x_upper_idx]]
        y_fit, y_lower, y_upper = y[[y_fit_idx, y_lower_idx, y_upper_idx]]

        x_estimate = x_fit, x_lower, x_upper
        y_estimate = y_fit, y_lower, y_upper

        dx_lower, dx_upper = x_fit - x_lower, x_upper - x_fit
        dy_lower, dy_upper = y_fit - y_lower, y_upper - y_fit

        # Plot estimates.
        x_panel.fill_between(
            x[x_lower_idx:(x_upper_idx + 1)],
            pdf_x[x_lower_idx:(x_upper_idx + 1)],
            antialiased=True, facecolor=[primary_colour],
            alpha=AREA_FILL_ALPHA, zorder=2
        )
        y_panel.fill_betweenx(
            y[y_lower_idx:(y_upper_idx + 1)],
            pdf_y[y_lower_idx:(y_upper_idx + 1)],
            antialiased=True, facecolor=[primary_colour],
            alpha=AREA_FILL_ALPHA, zorder=2
        )
    else:
        x_estimate, y_estimate = None, None

    return fig, x_estimate, y_estimate, patch


def plot_2d_constraints(chains, bins=None, smooth=None,
                        range_x=None, range_y=None, label_x='', label_y='',
                        estimate='median', precision_x=None, precision_y=None,
                        truth_x=None, truth_y=None, fig=None, figsize=None,
                        label=None, cmap=None, alpha=None,
                        show_estimates=True):
    """Plot 2-d parameter constraints from sample chains.

    Parameters
    ----------
    chains : float :class:`numpy.ndarray`
        Parameter chains.
    bins : int, array_like or None, optional
        Number of bins for (both) parameters (default is `None`).
    smooth : float or None, optional
        The standard deviation for Gaussian kernel passed to
        `scipy.ndimage.gaussian_filter` to smooth the posterior grid.
        If `None` (default), no smoothing is applied.
    range_x, range_y : tuple, optional
        Renormalisation range for the parameters (default is ()).
    label_x, label_y : str, optional
        Parameter name as a TeX string (default is '').
    estimate : {'maximum', 'median', None}, optional
        Parameter estimate type (default is 'median').
    precision_x, precision_y : int or None, optional
        Precision for the parameter estimate as a number of decimal places
        (default is `None`).
    truth_x, truth_y : float or None, optional
        Truth value for the parameter (default is `None`).
    fig : :class:`matplotlib.figure.Figure` *or None, optional*
        Any existing figures to plot on (default is `None`).
    figsize : tuple of float or None, optional
        Figure size in inches (default is `None`).
    label : (sequence of) str or None, optional
        Label for the parameter constraint (default is `None`).
    cmap : :class:`matplotlib.ScalarMappable` or None, optional
        Colour map for constraint contours (default is `None`).
    alpha : float or None, optional
        Transparency value for constraint contours (default is `None`).
    show_estimates : bool, optional
        If `True`, display the estimates if available.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Any existing figures to plot on.
    x_estimate, y_estimate : list of tuple
        Parameter estimate, lower uncertainty and upper uncertainty of
        for each likelihood value sets.

    """
    posterior, x, y = convert_chains_to_grid(
        chains, bins=bins, smooth=smooth, range_x=range_x, range_y=range_y
    )

    # Set up plottable areas.
    # pylint: disable=global-statement
    global legend_state
    if fig is None:
        fig = plt.figure("2-d constraint", figsize=figsize or (5.5, 5.5))
        canvas = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
        top_panel = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=canvas)
        side_panel = plt.subplot2grid((4, 4), (1, 3), rowspan=3, sharey=canvas)
        legend_state = ([], [])
    else:
        canvas, top_panel, side_panel = fig.axes

    try:
        cmap = ListedColormap(sns.color_palette(cmap))
    except (TypeError, ValueError):
        pass

    range_x = range_x or [np.min(x), np.max(x)]
    range_y = range_y or [np.min(y), np.max(y)]

    # Fill in plottable areas.
    fig, x_estimate, y_estimate, patch = plot_2d_contours(
        posterior, x, y, fig=fig, cmap=cmap, alpha=alpha,
        x_label=label_x, y_label=label_y, x_range=range_x, y_range=range_y,
        estimate=estimate, x_precision=precision_x, y_precision=precision_y
    )

    legend_state[0].append(patch)
    legend_state[1].append(label)

    if truth_x is not None:
        canvas.axvline(truth_x, c='k', ls='--', zorder=3)
    if truth_y is not None:
        canvas.axhline(truth_y, c='k', ls='--', zorder=3)

    # Adjust plottable areas.
    canvas.legend(*legend_state, fontsize='small')
    canvas.set_xlim(max(np.min(x), range_x[0]), min(np.max(x), range_x[-1]))
    canvas.set_ylim(max(np.min(y), range_y[0]), min(np.max(y), range_y[-1]))
    canvas.axes.tick_params(axis='x', which='both', direction='in', top=True)
    canvas.axes.tick_params(axis='y', which='both', direction='in', right=True)
    canvas.xaxis.set_minor_locator(AutoMinorLocator())
    canvas.yaxis.set_minor_locator(AutoMinorLocator())
    canvas.set_xlabel(r'${}$'.format(label_x), labelpad=8)
    canvas.set_ylabel(r'${}$'.format(label_y), labelpad=8)

    if show_estimates:
        top_panel.legend(
            bbox_to_anchor=[1.25, 0.775], loc='center',
            handlelength=1.25, labelspacing=0., fontsize='small'
        )
    top_panel.set_ylim(bottom=0)
    top_panel.axes.tick_params(
        axis='x', which='both', top=False, bottom=False, labelbottom=False
    )
    top_panel.axes.tick_params(
        axis='y', which='both', left=False, right=False, labelleft=False
    )
    top_panel.spines['top'].set_visible(False)
    top_panel.spines['left'].set_visible(False)
    top_panel.spines['right'].set_visible(False)

    if show_estimates:
        side_panel.legend(
            bbox_to_anchor=[0.75, 1.075],
            loc='center', handlelength=1.25, labelspacing=0., fontsize='small'
        )
    side_panel.set_xlim(left=0)
    side_panel.axes.tick_params(
        axis='x', which='both', top=False, bottom=False, labelbottom=False
    )
    side_panel.axes.tick_params(
        axis='y', which='both', left=False, right=False, labelleft=False
    )
    side_panel.spines['top'].set_visible(False)
    side_panel.spines['bottom'].set_visible(False)
    side_panel.spines['right'].set_visible(False)

    return fig, x_estimate, y_estimate
