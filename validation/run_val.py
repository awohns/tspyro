import argparse
import pickle

import tskit

import numpy as np
import torch
from pyro import poutine

from fit import fit_guide
from models import euclidean_migration, marginal_euclidean_migration
from analyze import compute_baselines
from util import get_metadata


def load_data(args):
    ts = tskit.load(args.ts).simplify()
    print("Tree sequence {} has {} nodes. Using Ne = {} and mu = {:.1e}.".format(args.ts, ts.num_samples, args.Ne, args.mu))
    return ts


def main(args):
    print(args)

    ts = load_data(args)

    result = {}

    if args.num_milestones == 0:
        milestones = None
    else:
        milestones = np.linspace(0, args.num_steps, args.num_milestones + 2)[1:-1]
        print("optim milestones: ", milestones)

    locations, true_times, is_leaf, is_internal, time_mask = get_metadata(ts, args)
    device = 'cpu' if args.device == 'cpu' else 'cuda'

    if args.time_init == 'prior':
        init_times = None
    elif args.time_init == 'tsdate':
        init_times = compute_baselines(args.ts, Ne=args.Ne, mu=args.mu,
                                       true_internal_times=ts.tables.nodes.time[is_internal],
                                       is_internal=is_internal)['tsdate_times'][is_internal]
        init_times = torch.from_numpy(init_times).to(dtype=torch.get_default_dtype(), device=device)
    elif args.time_init == 'truth':
        init_times = ts.tables.nodes.time[is_internal]
        init_times = torch.from_numpy(init_times).to(dtype=torch.get_default_dtype(), device=device)

    # Let's only infer times
    if args.migration == 'none':
        inferred_times, _, _, guide, losses, final_elbo = fit_guide(
            ts, leaf_location=None, Ne=args.Ne, mutation_rate=args.mu, steps=args.num_steps, log_every=args.log_every,
            learning_rate=args.init_lr, milestones=milestones, seed=args.seed, migration_likelihood=None,
            gamma=args.gamma, init_times=init_times, device=device,
            gap_prefactor=args.gap_prefactor, gap_exponent=args.gap_exponent)

    # Let's perform joint inference of time and location
    elif args.migration in ['euclidean', 'marginal_euclidean']:
        migration_likelihood = poutine.mask(marginal_euclidean_migration, mask=time_mask) if 'marg' in args.migration else \
            poutine.mask(euclidean_migration, mask=time_mask)
        leaf_locations = torch.from_numpy(locations).to(dtype=torch.get_default_dtype(), device=device)[is_leaf]

        inferred_times, inferred_locations, inferred_migration_scale, guide, losses, final_elbo = fit_guide(
            ts, leaf_location=leaf_locations, migration_likelihood=migration_likelihood,
            mutation_rate=args.mu, steps=args.num_steps, log_every=args.log_every, Ne=args.Ne,
            learning_rate=args.init_lr, milestones=milestones, seed=args.seed, gamma=args.gamma,
            init_times=init_times, device=device,
            gap_prefactor=args.gap_prefactor, gap_exponent=args.gap_exponent)

        inferred_internal_locations = inferred_locations[is_internal]

        result['inferred_migration_scale'] = inferred_migration_scale.item()
        result['inferred_internal_locations'] = inferred_internal_locations.data.cpu().numpy()

    result['inferred_internal_times'] = inferred_times.data.cpu().numpy()[is_internal]
    result['losses'] = losses
    result['Ne'] = args.Ne
    result['mu'] = args.mu
    result['time_cutoff'] = args.time_cutoff
    result['ts_filename'] = args.ts
    result['final_elbo'] = final_elbo
    result['migration'] = args.migration

    tag = '{}.tcut{}.s{}.numstep{}k.milestones{}_{}.tinit_{}.lr{}.gap_{}_{}.{}'
    tag = tag.format(args.migration, args.time_cutoff, args.seed, args.num_steps // 1000,
                     args.num_milestones, int(10 * args.gamma), args.time_init, int(1000 * args.init_lr),
                     int(10 * args.gap_prefactor), int(10 * args.gap_exponent), args.ts)
    f = args.out + 'result.{}.pkl'.format(tag)
    pickle.dump(result, open(f, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')
    default_ts = 'slim_2d_Ne_2000_continuous_recapitated_length_1e8_mutated_mu_1e8_rep_1.trees'
    parser.add_argument('--ts', type=str, default=default_ts)
    parser.add_argument('--out', type=str, default='./out/')
    parser.add_argument('--migration', type=str, default='none',
                        choices=['euclidean', 'marginal_euclidean', 'none'])
    parser.add_argument('--time', type=str, default='naive', choices=['naive'])
    parser.add_argument('--time-init', type=str, default='prior', choices=['prior', 'tsdate', 'truth'])
    parser.add_argument('--init-lr', type=float, default=0.05)
    parser.add_argument('--time-cutoff', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--gap-prefactor', type=float, default=5.0)
    parser.add_argument('--gap-exponent', type=float, default=1.0)
    parser.add_argument('--num-milestones', type=int, default=2)
    parser.add_argument('--Ne', type=int, default=2000)
    parser.add_argument('--mu', type=float, default=1.0e-8)
    parser.add_argument('--num-steps', type=int, default=45 * 1000)
    parser.add_argument('--log-every', type=int, default=3000)
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif args.device == 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)

    main(args)
