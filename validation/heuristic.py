import argparse
import tskit
import torch
import time
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


def get_metadata(ts):
    locations = []
    for node in ts.nodes():
        if node.individual != -1:
            loc = ts.individual(node.individual).location[:2]
            if len(loc) == 2:
                locations.append(ts.individual(node.individual).location[:2])
            else:
                locations.append(np.array([np.nan, np.nan]))
        else:
            locations.append(np.array([np.nan, np.nan]))

    is_leaf = np.array(ts.tables.nodes.flags & 1, dtype=bool)
    is_internal = ~is_leaf
    true_times = ts.tables.nodes.time

    return np.array(locations), true_times, is_leaf, is_internal


def load_data(args):
    t0 = time.time()
    ts = tskit.load(args.ts)
    t1 = time.time()
    print("Tree sequence {} has {} nodes. Took {:.2f} seconds to load.".format(args.ts, ts.num_samples, t1 - t0))
    return ts


def average_children(
    parent, edges,
    locations: np.ndarray,
) -> typing.Tuple[int, np.ndarray]:

    edges = list(edges)
    child_lats = list()
    child_longs = list()
    num_child_nans = 0

    for edge in edges:
        lat, lon = locations[edge.child]
        if not np.isnan(lat + lon):
            child_lats.append(lat)
            child_longs.append(lon)
        else:
            num_child_nans += 1

    #if num_child_nans > 0 and num_child_nans == len(edges):
    #    print("for parent {} at time {:.5f} we encountered {} children and {} nan child locations".format(parent,
    #          times[parent], len(edges), num_child_nans))

    val = np.average(np.array([child_lats, child_longs]).T, axis=0)
    return val


def average_parents(
    child, edges,
    locations: np.ndarray,
) -> typing.Tuple[int, np.ndarray]:

    edges = list(edges)
    parent_lats = list()
    parent_longs = list()
    num_parent_nans = 0

    for edge in edges:
        lat, lon = locations[edge.parent]
        if not np.isnan(lat + lon):
            parent_lats.append(lat)
            parent_longs.append(lon)
        else:
            num_parent_nans += 1

    if num_parent_nans > 0:
        print("num_parent_nans", num_parent_nans)

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


def main(args):
    print(args)

    ts = load_data(args)

    nodes = ts.tables.nodes
    edges = ts.tables.edges

    is_sample = torch.tensor((nodes.flags & 1).astype(bool), dtype=torch.bool)
    locations = torch.as_tensor(get_metadata(ts)[0])
    nan_locations = locations.isnan().sum(-1) > 0
    print("# nan locations among all nodes in ts: ", nan_locations.sum().item())

    observed_loc = (~nan_locations) & is_sample
    # assert that all non-nan locs are samples
    assert (observed_loc.float() - (~nan_locations).float()).abs().max().item() == 0.0
    unobserved_loc = ~observed_loc

    heuristic_loc = get_ancestral_geography(ts, locations, observed_loc)
    print("# nans in heuristic_loc returned by get_ancestral_geography: ",
          (heuristic_loc.isnan().sum(-1) > 0).sum().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')

    default_ts = 'ancient_chr20.dates_added.simplified.dated.trees'
    #default_ts = 'ancient_chr20.dates_added.simplified.dated.down_500_0.trees'
    default_ts = 'ancient_chr20.dates_added.simplified.dated.down_50_0.trees'

    parser.add_argument('--ts', type=str, default=default_ts)
    args = parser.parse_args()
    torch.set_default_tensor_type(torch.FloatTensor)

    main(args)
