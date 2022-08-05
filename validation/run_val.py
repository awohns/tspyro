import argparse
import pickle

import tspyro
import tskit
import tsdate

import numpy as np
import torch
from pyro import poutine

from fit import fit_guide


def load_data(filename):
    ts = tskit.load(filename).simplify()
    print("Tree sequence has {} nodes".format(ts.num_samples))
    return ts


def get_leaf_locations(ts):
    # Next, we need to get the locations of the nodes
    locations = []
    for node in ts.nodes():
        if node.individual != -1:
            locations.append(ts.individual(node.individual).location[:2])
        else:
            locations.append(np.array([np.nan, np.nan]))

    locations = np.array(locations)
    internal_locations = locations[ts.num_samples:]
    leaf_locations = torch.from_numpy(locations[:ts.num_samples])

    return leaf_locations, internal_locations


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

    ts = load_data(args.ts)
    priors = tsdate.build_prior_grid(ts, Ne=args.Ne)

    result = {}

    if args.num_milestones == 0:
        milestones = None
    else:
        milestones = np.linspace(0, args.num_steps, args.num_milestones + 2)[1:-1]
        print("optim milestones: ", milestones)

    if args.model == 'time':  # Let's only infer times
        inferred_times, _, _, guide, losses, final_elbo = fit_guide(
            ts, leaf_location=None, priors=priors, mutation_rate=1e-8, steps=args.num_steps, log_every=args.log_every,
            learning_rate=args.init_lr, milestones=milestones, seed=args.seed, migration_likelihood=None)

    elif args.model == 'joint':  # Let's perform joint inference of time and location
        leaf_locations, internal_locations = get_leaf_locations(ts)

        mask = get_time_mask(ts, args)
        migration_likelihood = tspyro.models.euclidean_migration
        migration_likelihood = poutine.mask(migration_likelihood, mask=mask)

        inferred_times, inferred_locations, inferred_migration_scale, guide, losses, final_elbo = fit_guide(
            ts, leaf_location=leaf_locations, migration_likelihood=migration_likelihood,
            priors=priors, mutation_rate=1e-8, steps=args.num_steps, log_every=args.log_every,
            learning_rate=args.init_lr, milestones=milestones, seed=args.seed)

        inferred_internal_locations = inferred_locations[ts.num_samples:]

        result['inferred_migration_scale'] = inferred_migration_scale.item()
        result['inferred_internal_locations'] = inferred_internal_locations.data.cpu().numpy()
        result['true_internal_locations'] = internal_locations

    result['inferred_times'] = inferred_times.data.cpu().numpy()
    result['inferred_internal_times'] = inferred_times.data.cpu().numpy()[ts.num_samples:]
    result['losses'] = losses
    result['true_times'] = ts.tables.nodes.time
    result['true_internal_times'] = ts.tables.nodes.time[ts.num_samples:]
    result['Ne'] = args.Ne
    result['time_cutoff'] = args.time_cutoff
    result['ts_filename'] = args.ts
    result['final_elbo'] = final_elbo
    result['model'] = args.model

    assert result['true_times'].shape == result['inferred_times'].shape
    if 'inferred_internal_locations' in result:
        assert result['inferred_internal_locations'].shape == result['inferred_internal_locations'].shape

    tag = '{}.tcut{}.s{}.Ne{}.numstep{}k.milestones{}.lr{}'
    tag = tag.format(args.model, args.time_cutoff, args.seed, args.Ne, args.num_steps // 1000,
                     args.num_milestones, int(1000 * args.init_lr))
    f = args.out + 'result.{}.pkl'.format(tag)
    pickle.dump(result, open(f, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')
    parser.add_argument('--ts', type=str, default='slim_2d_continuous_recapitated_mutated.down_200_0.trees')
    parser.add_argument('--out', type=str, default='./out/')
    parser.add_argument('--model', type=str, default='time', choices=['time', 'space', 'joint'])
    parser.add_argument('--init-lr', type=float, default=0.05)
    parser.add_argument('--time-cutoff', type=float, default=200.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-milestones', type=int, default=3)
    parser.add_argument('--Ne', type=int, default=1000)
    parser.add_argument('--num-steps', type=int, default=3000)
    parser.add_argument('--log-every', type=int, default=2000)
    args = parser.parse_args()

    main(args)
