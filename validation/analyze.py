import argparse
import pickle

import tspyro
import tskit
import tsdate

import numpy as np
import torch
from pyro import poutine

from fit import fit_guide
from run_val import load_data

from sklearn.metrics import mean_squared_error, mean_squared_log_error


def get_leaf_locations(ts):
    # Next, we need to get the locations of the nodes
    node_locations = np.full((ts.num_nodes, 2), -1, dtype=float)
    no_location_nodes = []
    for node in ts.nodes():
        if node.individual != -1:
            node_locations[node.id, :] = ts.individual(node.individual).location[:2]
        else:
            no_location_nodes.append(node.id)
    # We want a tensor with only the leaf nodes for inference
    leaf_locations = torch.from_numpy(node_locations[ts.samples()])
    return leaf_locations


def get_time_mask(ts, args):
    # We restrict inference of locations to nodes within time_cutoff generations of the samples (which are all at time zero)
    masked = ts.tables.nodes.time >= args.time_cutoff
    mask = torch.ones(ts.num_edges, dtype=torch.bool)
    for e, edge in enumerate(ts.edges()):
        if masked[edge.child]:
            mask[e] = False

    return mask


def main(args):
    print(args)

    ts = load_data(args)
    priors = tsdate.build_prior_grid(ts, Ne=args.Ne)

    result = pickle.load(open(args.pkl, 'rb'))

    dated = tsdate.date(ts, mutation_rate=1e-8, Ne=args.Ne)
    tsdate_times = dated.tables.nodes.time

    inferred_times = result['inferred_times']
    true_times = result['true_times']

    pyro_time_msle = mean_squared_log_error(inferred_times, true_times)
    tsdate_time_msle = mean_squared_log_error(tsdate_times, true_times)

    print("pyro_time_msle", pyro_time_msle, "tsdate_time_msle", tsdate_time_msle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze validation run')
    parser.add_argument('--pkl', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-nodes', type=int, default=800)
    parser.add_argument('--ts', type=str, default='slim_2d_continuous_recapitated_mutated.trees')
    parser.add_argument('--Ne', type=int, default=1000)
    args = parser.parse_args()

    main(args)
