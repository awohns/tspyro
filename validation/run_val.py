import argparse
import pickle

import tspyro
import tskit
import tsdate

import numpy as np
import torch
from pyro import poutine

from fit import fit_guide


def load_data(args):
    ts = tskit.load(args.ts)

    if args.num_nodes is not None:
        assert args.num_nodes <= ts.num_samples
        if args.num_nodes < ts.num_samples:
            # Now let's randomly sample num_nodes leaf nodes
            np.random.seed(args.seed)
            random_sample = np.random.choice(np.arange(0, ts.num_samples), args.num_nodes, replace=False)
            sampled_ts = ts.simplify(samples=random_sample)
        print("Downsampled nodes: {} -> {}".format(ts.num_samples, args.num_nodes))
        return sampled_ts
    else:
        print("Tree sequence has {} nodes".format(ts.num_samples))
        return ts.simplify()


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

    result = {}

    if args.num_milestones == 0:
        milestones = None
    else:
        milestones = np.linspace(0, args.num_steps, args.num_milestones + 2)[1:-1]
        print("optim milestones: ", milestones)

    if args.model == 'time':  # Let's only infer times
        inferred_times, _, _, guide, losses = fit_guide(
            ts, leaf_location=None, priors=priors, mutation_rate=1e-8, steps=args.num_steps, log_every=args.log_every,
            learning_rate=args.init_lr, milestones=milestones, seed=args.seed, migration_likelihood=None)

    elif args.model == 'joint':  # Let's perform joint inference of time and location
        leaf_locations = get_leaf_locations(ts)

        mask = get_time_mask(ts, args)
        migration_likelihood = tspyro.models.euclidean_migration
        migration_likelihood = poutine.mask(migration_likelihood, mask=mask)

        inferred_times, inferred_locations, inferred_migration_scale, guide, losses = fit_guide(
            ts, leaf_location=leaf_locations, migration_likelihood=migration_likelihood,
            priors=priors, mutation_rate=1e-8, steps=args.num_steps, log_every=args.log_every,
            learning_rate=args.init_lr, milestones=milestones, seed=args.seed)

        result['inferred_locations'] = inferred_locations.data.cpu().numpy()
        result['inferred_migration_scale'] = inferred_migration_scale.item()

    result['inferred_times'] = inferred_times.data.cpu().numpy()
    result['losses'] = losses
    result['true_times'] = ts.tables.nodes.time
    result['Ne'] = args.Ne
    result['time_cutoff'] = args.time_cutoff

    tag = '{}.nodes{}.tcut{}.s{}.Ne{}.numstep{}.milestones{}'
    tag = tag.format(args.model, args.num_nodes, args.time_cutoff, args.seed, args.Ne, args.num_steps, args.num_milestones)
    f = args.out + 'result.{}.pkl'.format(tag)
    pickle.dump(result, open(f, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')
    parser.add_argument('--ts', type=str, default='slim_2d_continuous_recapitated_mutated.trees')
    parser.add_argument('--num-nodes', type=int, default=800)
    parser.add_argument('--out', type=str, default='./out/')
    parser.add_argument('--model', type=str, default='time', choices=['time', 'space', 'joint'])
    parser.add_argument('--init-lr', type=float, default=0.01)
    parser.add_argument('--time-cutoff', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-milestones', type=int, default=0)
    parser.add_argument('--Ne', type=int, default=1000)
    parser.add_argument('--num-steps', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=1000)
    args = parser.parse_args()

    main(args)