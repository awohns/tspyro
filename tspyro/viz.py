import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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


def plot_edges(ts, real_locs, inferred_locations):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

    # Plot the true locations
    ax[0].scatter(
        real_locs[:, 0], real_locs[:, 1], alpha=0.9, c="red", label="True location"
    )

    # Plot the inferred locations
    ax[1].scatter(
        inferred_locations[:, 0],
        inferred_locations[:, 1],
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
    x_pos, y_pos = edge_relationships(ts, inferred_locations)
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


def plot_diff(ts, real_locs, inferred_locations, ax, waypoints=None, title=None):

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
        inferred_locations[:, 0],
        inferred_locations[:, 1],
        alpha=0.5,
        s=0.5,
        c="blue",
        label="Inferred location",
    )

    x_pos = []
    y_pos = []
    for real, inferred in zip(real_locs, inferred_locations):
        x_pos.append(real[0])
        x_pos.append(inferred[0])
        x_pos.append(None)
        y_pos.append(real[1])
        y_pos.append(inferred[1])
        y_pos.append(None)

    ax.plot(x_pos, y_pos, alpha=0.2, lw=0.5)  # , label="Edge relationship")

    if title is None:
        ax.set_title("Differences", fontsize=20)
    else:
        ax.set_title(title, fontsize=20)

    ax.axis("equal")

    ax.legend(loc="lower right", fontsize=15)
