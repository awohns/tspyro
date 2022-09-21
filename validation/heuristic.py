import tskit
import torch
import numpy as np
import typing
import itertools
import operator


def edges_by_parent_asc(
    ts: tskit.TreeSequence
) -> typing.Iterable[typing.Tuple[int, typing.Iterable[tskit.Edge]]]:
    """
    Return an itertools.groupby object of edges grouped by parent in ascending order
    of the time of the parent. Since tree sequence properties guarantee that edges
    are listed in nondecreasing order of parent time
    (https://tskit.readthedocs.io/en/latest/data-model.html#edge-requirements)
    we can simply use the standard edge order
    """
    return itertools.groupby(ts.edges(), operator.attrgetter("parent"))


def edges_by_child_desc(
    ts: tskit.TreeSequence
) -> typing.Iterable[typing.Tuple[int, typing.Iterable[tskit.Edge]]]:
    """
    """
    return itertools.groupby(list(ts.edges())[::-1], operator.attrgetter("child"))


def average_children(
    parent, edges,
    locations: np.ndarray,
) -> typing.Tuple[int, np.ndarray]:

    edges = list(edges)
    child_lats = list()
    child_longs = list()

    for edge in edges:
        lat, lon = locations[edge.child]
        if not np.isnan(lat + lon):
            child_lats.append(lat)
            child_longs.append(lon)

    val = np.average(np.array([child_lats, child_longs]).T, axis=0)
    return val


def average_parents(
    child, edges,
    locations: np.ndarray,
) -> typing.Tuple[int, np.ndarray]:

    edges = list(edges)
    parent_lats = list()
    parent_longs = list()

    for edge in edges:
        lat, lon = locations[edge.parent]
        if not np.isnan(lat + lon):
            parent_lats.append(lat)
            parent_longs.append(lon)

    val = np.average(np.array([parent_lats, parent_longs]).T, axis=0)
    return val


def get_ancestral_geography(
    ts: tskit.TreeSequence,
    sample_locations: np.ndarray,
    observed: np.ndarray,
) -> torch.Tensor:

    locations = np.full((ts.num_nodes, 2), np.nan)
    locations[observed] = sample_locations[observed]
    fixed_nodes = set(np.arange(ts.num_nodes)[observed])

    num_child_nans = 0
    num_parent_nans = 0

    for parent, edges in edges_by_parent_asc(ts):
        if parent not in fixed_nodes:  # this should always be true, right?
            val = average_children(parent, edges, locations)
            locations[parent] = val
            if np.isnan(val[0]):
                num_child_nans += 1

    for child, edges in edges_by_child_desc(ts):
        if np.isnan(locations[child][0]):
            val = average_parents(child, edges, locations)
            locations[child] = val
            if np.isnan(val[0]):
                num_parent_nans += 1

    print("# of nans returned by average_children in edges_by_parent_asc loop: ", num_child_nans)
    print("# of nans returned by average_parents in edges_by_parent_desc loop: ", num_parent_nans)

    return torch.tensor(
        locations, dtype=torch.get_default_dtype()  # noqa: E203
        )
