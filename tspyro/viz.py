import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse


def edge_relationships(ts, loc_arr):
    x_pos = []
    y_pos = []
    for parent, child in zip(ts.tables.edges.parent, ts.tables.edges.child):
        x_pos.append(loc_arr[parent][0])
        x_pos.append(loc_arr[child][0])
        x_pos.append(None)
        y_pos.append(loc_arr[parent][1])
        y_pos.append(loc_arr[child][1])
        y_pos.append(None)
    return x_pos, y_pos


def plot_edges(ts, real_locs, inferred_locs):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

    # Plot the true locations
    ax[0].scatter(
        real_locs[:, 0], real_locs[:, 1], alpha=0.9, c="red", label="True location"
    )

    # Plot the inferred locations
    ax[1].scatter(
        inferred_locs[:, 0],
        inferred_locs[:, 1],
        alpha=0.9,
        c="blue",
        label="Inferred location",
    )

    # Real gaps
    real_gaps = (
        ts.tables.nodes.time[ts.tables.edges.parent]
        - ts.tables.nodes.time[ts.tables.edges.child]
    )

    x_pos, y_pos = edge_relationships(ts, real_locs)
    ax[0].plot(x_pos, y_pos, alpha=0.2)  # , label="Edge relationship")
    x_pos, y_pos = edge_relationships(ts, inferred_locs)
    ax[1].plot(x_pos, y_pos, alpha=0.2)  # , label="Edge relationship")
    cmap = matplotlib.cm.get_cmap("inferno")
    cmap(real_gaps / np.max(real_gaps))

    def label_point(l_x, l_y, l_val, ax):
        for x, y, val in zip(l_x, l_y, l_val):
            ax.text(x, y, str(val))

    ax[0].set_title("True Locations", fontsize=20)
    ax[1].set_title("tspyro", fontsize=20)

    ax[0].axis("equal")
    ax[1].axis("equal")

    plt.legend(loc="lower right", fontsize=15)
    plt.tight_layout()


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="lightgray", **kwargs):
    """
    From https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        alpha=0.5,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_diff(
    ts,
    real_locs,
    inferred_locs,
    ax,
    title=None,
    waypoints=None,
    samples=None,
    confidence_ellipse_std=None,
    **kwargs
):
    """
    Plot the difference between true and inferred locations.

    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`
    :param torch.tensor real_locs: A 2d tensor of the real locations of
        each node in the tree sequence
    :param torch.tensor inferred_locs: A 2d tensor of the inferred locations
        of each node in the tree sequence
    :param matplotlib.axes.Axes ax: A matplotlib axis on which to plot
        real and inferred locations
    :param string title: If not `None`, displays a title for the figure.
        Default: None.
    :param torch.tensor waypoints: The locations of waypoints to be plotted.
        Default=None.
    :param torch.tensor samples: Random draws from the posterior distribution
        on locations, used to plot confidence ellipses around each node.
        The second dimension of this tensor must match `real_locs` and
        `inferred_locs`. Warning: this can be slow if many nodes are being plotted.
        Default=False.
    :param int confidence_ellipse_std: If `confidence_ellipse` is `True`, controls
        the number of standard deviations to determine the ellipse's radiuses.
    :param \\**kwargs: All further keyword arguments are passed to
        `~matplotlib.patches.Ellipse` command.
    """
    if real_locs.shape != inferred_locs.shape:
        raise ValueError("`real_locs` and `inferred_locs` must have the same shape")

    if samples is not None:
        if samples.shape[1] + ts.num_samples != real_locs.shape[0]:
            raise ValueError(
                "Second and third dimensions of `samples` must match "
                "`real_locs` and `inferred_locs`"
            )
        if confidence_ellipse_std is None:
            raise ValueError(
                "Must specify `confidence_ellipse_std` if providing `samples`"
            )

    if confidence_ellipse_std is not None and samples is None:
        raise ValueError(
            "Cannot provide a value for `confidence_ellipse_std` ifi "
            "`confidence_ellipse` is `False`"
        )

    if waypoints is not None:
        ax.scatter(waypoints[:, 0], waypoints[:, 1], c="black", alpha=0.1, zorder=-1)

    # Plot the true locations
    ax.scatter(
        real_locs[:, 0],
        real_locs[:, 1],
        alpha=0.5,
        s=0.5,
        c="red",
        label="True location",
    )

    # Plot the inferred locations
    ax.scatter(
        inferred_locs[:, 0],
        inferred_locs[:, 1],
        alpha=0.5,
        s=0.5,
        c="blue",
        label="Inferred location",
    )

    x_pos = []
    y_pos = []
    for real, inferred in zip(real_locs, inferred_locs):
        x_pos.append(real[0])
        x_pos.append(inferred[0])
        x_pos.append(None)
        y_pos.append(real[1])
        y_pos.append(inferred[1])
        y_pos.append(None)

    ax.plot(x_pos, y_pos, alpha=0.2, lw=0.5)  # , label="Edge relationship")

    if samples is not None:
        for i in range(0, samples.shape[1]):
            confidence_ellipse(
                samples[:, i, 0].numpy(),
                samples[:, i, 1].numpy(),
                ax=ax,
                n_std=confidence_ellipse_std,
                **kwargs
            )

    if title is None:
        ax.set_title("Differences", fontsize=20)
    else:
        ax.set_title(title, fontsize=20)

    ax.axis("equal")

    ax.legend(loc="lower right", fontsize=15)
