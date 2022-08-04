import argparse

import tspyro
import tskit
import tsdate

import numpy as np
import torch
from pyro import poutine


def load_data(args):
    ts = tskit.load(args.ts)

    # Now let's randomly sample 200 leaf nodes
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(0, ts.num_samples), 200, replace=False)
    sampled_ts = ts.simplify(samples=random_sample)

    return sampled_ts

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

    leaf_locations = get_leaf_locations(ts)

    mask = get_time_mask(ts, args)
    migration_likelihood = tspyro.models.euclidean_migration
    migration_likelihood = poutine.mask(migration_likelihood, mask=mask)

    # Let's perform joint inference of time and location
    priors = tsdate.build_prior_grid(ts, Ne=args.Ne)
    inferred_times_joint, inferred_locations, migration_scale_joint, guide_joint, losses_joint = \
        tspyro.models.fit_guide(ts, leaf_location=leaf_locations,
                                migration_likelihood=migration_likelihood,
                                priors=priors, mutation_rate=1e-8, steps=args.num_steps, log_every=args.log_every)

    print(inferred_times_joint.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')
    parser.add_argument('--ts', type=str, default='slim_2d_continuous_recapitated_mutated.trees')
    parser.add_argument('--out', type=str, default='./out/')
    parser.add_argument('--model', type=str, default='time', choices=['time', 'space', 'joint'])
    parser.add_argument('--init-lr', type=float, default=0.005)
    parser.add_argument('--time-cutoff', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--Ne', type=int, default=1000)
    parser.add_argument('--num-steps', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=1000)
    args = parser.parse_args()

    main(args)
