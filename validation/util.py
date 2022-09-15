import tskit
import numpy as np
import torch
import pyslim
import typing
from tspyro.ops import edges_by_parent_asc, average_edges


def get_ancestral_geography(
    ts: tskit.TreeSequence,
    sample_locations: np.ndarray,
    observed: np.ndarray,
    show_progress: typing.Optional[bool] = True
) -> torch.Tensor:
    locations = np.zeros((ts.num_nodes, 2))
    #locations[ts.samples()] = sample_locations
    locations[observed] = sample_locations[observed]
    #fixed_nodes = set(ts.samples())
    fixed_nodes = set(np.arange(ts.num_nodes)[observed])
    is_internal = ~np.array((ts.tables.nodes.flags & 1).astype(bool), dtype=bool)
    # Iterate through the nodes via groupby on parent node
    for parent_edges in edges_by_parent_asc(ts):
        if parent_edges[0] not in fixed_nodes:
            parent, val = average_edges(parent_edges, locations)
            locations[parent] = val
    return torch.tensor(
        locations, dtype=torch.get_default_dtype()  # noqa: E203
        #locations[is_internal], dtype=torch.get_default_dtype()  # noqa: E203
        )


def downsample_ts(filename, num_nodes, seed):
    ts = tskit.load(filename)
    np.random.seed(seed)
    random_sample = np.random.choice(np.arange(0, ts.num_samples), num_nodes, replace=False)
    sampled_ts = ts.simplify(samples=random_sample)
    sampled_ts.dump('.'.join(filename.split('.')[:-1]) + '.down_{}_{}.trees'.format(num_nodes, seed))


def recap_ts(filename, ancestral_Ne):
    ts = tskit.load(filename)
    recapitated_ts = pyslim.recapitate(pyslim.update(ts), ancestral_Ne=ancestral_Ne)
    recapitated_ts.dump('.'.join(filename.split('.')[:-1]) + '.recap.trees')


def get_time_mask(ts, time_cutoff, times):
    # We restrict inference of locations to nodes within time_cutoff generations of the samples (which are all at time zero)
    masked = times > time_cutoff
    mask = torch.ones(ts.num_edges, dtype=torch.bool)
    for e, edge in enumerate(ts.edges()):
        if masked[edge.child]:
            mask[e] = False

    return mask


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
