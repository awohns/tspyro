import argparse
import pickle

import tskit

import numpy as np
import torch
from pyro import poutine

from fit import fit_guide
from models import euclidean_migration, marginal_euclidean_migration
from analyze import compute_baselines


def load_data(filename):
    ts = tskit.load(filename).simplify()
    print("Tree sequence has {} nodes".format(ts.num_samples))
    return ts


def get_time_mask(ts, args):
    # We restrict inference of locations to nodes within time_cutoff generations of the samples (which are all at time zero)
    masked = ts.tables.nodes.time >= args.time_cutoff
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
    time_mask = get_time_mask(ts, args)

    return np.array(locations), is_leaf, is_internal, time_mask


def main(args):
    print(args)

    ts = load_data(args.ts)

    result = {}

    if args.num_milestones == 0:
        milestones = None
    else:
        milestones = np.linspace(0, args.num_steps, args.num_milestones + 2)[1:-1]
        print("optim milestones: ", milestones)

    locations, is_leaf, is_internal, time_mask = get_metadata(ts, args)

    if args.time_init == 'prior':
        init_times = None
    elif args.time_init == 'tsdate':
        init_times = compute_baselines(args.ts, Ne=args.Ne,
                                       true_internal_times=ts.tables.nodes.time[is_internal],
                                       is_internal=is_internal)['tsdate_times'][is_internal]
        init_times = torch.from_numpy(init_times).to(dtype=torch.get_default_dtype())
    elif args.time_init == 'truth':
        init_times = ts.tables.nodes.time[is_internal]
        init_times = torch.from_numpy(init_times).to(dtype=torch.get_default_dtype())

    # Let's only infer times
    if args.migration == 'none':
        inferred_times, _, _, guide, losses, final_elbo = fit_guide(
            ts, leaf_location=None, Ne=args.Ne, mutation_rate=args.mu, steps=args.num_steps, log_every=args.log_every,
            learning_rate=args.init_lr, milestones=milestones, seed=args.seed, migration_likelihood=None,
            gamma=args.gamma, init_times=init_times)

    # Let's perform joint inference of time and location
    elif args.migration in ['euclidean', 'marginal_euclidean']:
        migration_likelihood = poutine.mask(marginal_euclidean_migration, mask=time_mask) if 'marg' in args.migration else \
            poutine.mask(euclidean_migration, mask=time_mask)
        leaf_locations = torch.from_numpy(locations).to(dtype=torch.get_default_dtype())[is_leaf]

        inferred_times, inferred_locations, inferred_migration_scale, guide, losses, final_elbo = fit_guide(
            ts, leaf_location=leaf_locations, migration_likelihood=migration_likelihood,
            mutation_rate=args.mu, steps=args.num_steps, log_every=args.log_every, Ne=args.Ne,
            learning_rate=args.init_lr, milestones=milestones, seed=args.seed, gamma=args.gamma,
            init_times=init_times)

        inferred_internal_locations = inferred_locations[is_internal]

        result['inferred_migration_scale'] = inferred_migration_scale.item()
        result['inferred_internal_locations'] = inferred_internal_locations.data.cpu().numpy()
        result['true_internal_locations'] = locations[is_internal]

    result['inferred_times'] = inferred_times.data.cpu().numpy()
    result['inferred_internal_times'] = inferred_times.data.cpu().numpy()[is_internal]
    result['losses'] = losses
    result['true_times'] = ts.tables.nodes.time
    result['true_internal_times'] = ts.tables.nodes.time[is_internal]
    result['Ne'] = args.Ne
    result['time_cutoff'] = args.time_cutoff
    result['ts_filename'] = args.ts
    result['final_elbo'] = final_elbo
    result['migration'] = args.migration
    result['is_leaf'] = is_leaf
    result['is_internal'] = is_internal

    assert result['true_times'].shape == result['inferred_times'].shape
    if 'inferred_internal_locations' in result:
        assert result['inferred_internal_locations'].shape == result['inferred_internal_locations'].shape

    tag = '{}.tcut{}.s{}.Ne{}.numstep{}k.milestones{}_{}.tinit_{}.lr{}.{}'
    tag = tag.format(args.migration, args.time_cutoff, args.seed, args.Ne, args.num_steps // 1000,
                     args.num_milestones, int(10 * args.gamma), args.time_init, int(1000 * args.init_lr), args.ts)
    f = args.out + 'result.{}.pkl'.format(tag)
    pickle.dump(result, open(f, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')
    default_ts = 'slim_2d_continuous_recapitated_mutated.down_200_0.trees'
    parser.add_argument('--ts', type=str, default=default_ts)
    parser.add_argument('--out', type=str, default='./out/')
    parser.add_argument('--migration', type=str, default='none',
                        choices=['euclidean', 'marginal_euclidean', 'none'])
    parser.add_argument('--time', type=str, default='naive', choices=['naive'])
    parser.add_argument('--time-init', type=str, default='tsdate', choices=['prior', 'tsdate', 'truth'])
    parser.add_argument('--init-lr', type=float, default=0.05)
    parser.add_argument('--time-cutoff', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--num-milestones', type=int, default=3)
    parser.add_argument('--Ne', type=int, default=2000)
    parser.add_argument('--mu', type=float, default=1.0e-7)
    parser.add_argument('--num-steps', type=int, default=40000)
    parser.add_argument('--log-every', type=int, default=2000)
    args = parser.parse_args()

    main(args)
