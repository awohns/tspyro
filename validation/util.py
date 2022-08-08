import tskit
import numpy as np
import torch


def downsample_ts(filename, num_nodes, seed):
    ts = tskit.load(filename)
    np.random.seed(seed)
    random_sample = np.random.choice(np.arange(0, ts.num_samples), num_nodes, replace=False)
    sampled_ts = ts.simplify(samples=random_sample)
    sampled_ts.dump(filename.split('.')[0] + '.down_{}_{}.trees'.format(num_nodes, seed))


def get_time_mask(ts, time_cutoff, times):
    # We restrict inference of locations to nodes within time_cutoff generations of the samples (which are all at time zero)
    masked = times >= time_cutoff
    mask = torch.ones(ts.num_edges, dtype=torch.bool)
    for e, edge in enumerate(ts.edges()):
        if masked[edge.child]:
            mask[e] = False

    return mask


def get_metadata(ts, args):
    locations = []
    for node in ts.nodes():
        if node.individual != -1:
            locations.append(ts.individual(node.individual).location[:2])
        else:
            locations.append(np.array([np.nan, np.nan]))

    is_leaf = np.array(ts.tables.nodes.flags & 1, dtype=bool)
    is_internal = ~is_leaf
    true_times = ts.tables.nodes.time

    return np.array(locations), true_times, is_leaf, is_internal
