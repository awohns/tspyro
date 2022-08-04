import argparse

import tspyro
import tskit
import tsdate

import numpy as np
import torch
from pyro import poutine


def main(args):
    print(args)

    ts = tskit.load(args.ts)

    # Now let's randomly sample 200 leaf nodes
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(0, ts.num_samples), 200, replace=False)
    sampled_ts = ts.simplify(samples=random_sample)

    # Next, we need to get the locations of the nodes
    node_locations = np.full((sampled_ts.num_nodes, 2), -1, dtype=float)
    no_location_nodes = []
    for node in sampled_ts.nodes():
        if node.individual != -1:
            node_locations[node.id, :] = sampled_ts.individual(node.individual).location[:2]
        else:
            no_location_nodes.append(node.id)
    # We want a tensor with only the leaf nodes for inference
    leaf_locations = torch.from_numpy(node_locations[sampled_ts.samples()])

    # Let's perform joint inference of time and location
    # We restrict inference of locations to nodes within time_cutoff generations of the samples (which are all at time zero)
    masked = sampled_ts.tables.nodes.time >= args.time_cutoff
    mask = torch.ones(sampled_ts.num_edges, dtype=torch.bool)
    for e, edge in enumerate(sampled_ts.edges()):
        if masked[edge.child]:
            mask[e] = False

    migration_likelihood = tspyro.models.euclidean_migration
    migration_likelihood = poutine.mask(migration_likelihood, mask=mask)

    priors = tsdate.build_prior_grid(sampled_ts, Ne=1000)
    inferred_times_joint, inferred_locations, migration_scale_joint, guide_joint, losses_joint = \
        tspyro.models.fit_guide(sampled_ts, leaf_location=leaf_locations,
                                migration_likelihood=migration_likelihood,
                                priors=priors, mutation_rate=1e-8, steps=args.num_steps, log_every=1000)

    print(inferred_times_joint.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')
    parser.add_argument('--ts', type=str, default='slim_2d_continuous_recapitated_mutated.trees')
    parser.add_argument('--init-lr', type=float, default=0.005)
    parser.add_argument('--time-cutoff', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=100)
    args = parser.parse_args()

    main(args)
