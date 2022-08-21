import argparse
import pickle

import tskit

import numpy as np
import torch
from pyro import poutine

from fit import fit_guide
from models import euclidean_migration, marginal_euclidean_migration, ConditionedTimesNaiveModel
from analyze import compute_baselines
from util import get_metadata, get_time_mask


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
        milestones = [int(x) for x in np.linspace(0, args.num_steps, args.num_milestones + 2)[1:-1]]
        print("optim milestones: ", milestones)

    locations, true_times, is_leaf, is_internal = get_metadata(ts)

    device = 'cpu' if args.device == 'cpu' else 'cuda'

    if args.time_init == 'tsdate':
        #raise NotImplementedError
        init_times = compute_baselines(args.ts, Ne=args.Ne, mu=args.mu)[0]['tsdate_times'][is_internal]
        init_times = torch.from_numpy(init_times).to(dtype=torch.get_default_dtype(), device=device)
        time_mask = get_time_mask(ts, args.time_cutoff, init_times)
    elif args.time_init == 'truth':
        true_times = ts.tables.nodes.time
        true_times = torch.from_numpy(true_times).to(dtype=torch.get_default_dtype(), device=device)
        init_times = ts.tables.nodes.time[is_internal]
        init_times = torch.from_numpy(init_times).to(dtype=torch.get_default_dtype(), device=device)
        time_mask = get_time_mask(ts, args.time_cutoff, true_times)

    migration_likelihood = poutine.mask(marginal_euclidean_migration, mask=time_mask) if 'marg' in args.migration else \
        poutine.mask(euclidean_migration, mask=time_mask)
    leaf_locations = torch.from_numpy(locations).to(dtype=torch.get_default_dtype(), device=device)[is_leaf]

    inferred_times, inferred_locations, inferred_migration_scale, guide, losses, final_elbo = fit_guide(
        ts, leaf_location=leaf_locations, migration_likelihood=migration_likelihood,
        mutation_rate=args.mu, steps=args.num_steps, log_every=args.log_every, Ne=args.Ne,
        learning_rate=args.init_lr, milestones=milestones, seed=args.seed, gamma=args.gamma,
        Model=ConditionedTimesNaiveModel, init_times=init_times, device=device, inference=args.inference,
        gap_prefactor=args.gap_prefactor, gap_exponent=args.gap_exponent, num_particles=args.num_particles,
        time_mask=time_mask, time_cutoff=args.time_cutoff)

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

    tag = '{}.tcut{}.s{}.numstep{}k.np{}.milestones{}_{}.tinit_{}.lr{}.gap_{}_{}_{}.{}.{}'
    tag = tag.format(args.migration, args.time_cutoff, args.seed, args.num_steps // 1000,
                     args.num_particles,
                     args.num_milestones, int(10 * args.gamma), args.time_init, int(1000 * args.init_lr),
                     int(10 * args.gap_prefactor), int(10 * args.gap_exponent), int(10 * args.min_gap),
                     args.inference, args.ts)
    f = args.out + 'result.{}.pkl'.format(tag)
    pickle.dump(result, open(f, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')

    default_ts = \
        ['slim_cont_nocomp_N_2e4_Ne_1e4_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8000_ancs_70_rep_1.recap.trees',
    'slim_cont_nocomp_N_2e4_Ne_1e4_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8000_ancs_7_rep_1.recap.trees',
    'slim_cont_nocomp_N_2e4_Ne_8.5e3_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8000_ancs_79_rep_1.recap.trees',
    'slim_cont_nocomp_N_2e4_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8e4_ancs_790_rep_1.recap.trees']
    default_ne = [10000, 10000, 8500, 10000]
    which = 2
    default_ts = default_ts[which]
    default_ne = default_ne[which]

    parser.add_argument('--ts', type=str, default=default_ts)
    parser.add_argument('--out', type=str, default='./space/')
    parser.add_argument('--migration', type=str, default='marginal_euclidean',
                        choices=['euclidean', 'marginal_euclidean'])
    parser.add_argument('--time-init', type=str, default='truth', choices=['tsdate', 'truth'])
    parser.add_argument('--inference', type=str, default='svi', choices=['svi', 'map', 'svilowrank'])
    parser.add_argument('--init-lr', type=float, default=0.1)
    parser.add_argument('--time-cutoff', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--gap-prefactor', type=float, default=1.0)
    parser.add_argument('--gap-exponent', type=float, default=1.0)
    parser.add_argument('--min-gap', type=float, default=1.0)
    parser.add_argument('--num-milestones', type=int, default=4)
    parser.add_argument('--Ne', type=int, default=default_ne)
    parser.add_argument('--mu', type=float, default=1.0e-8)
    parser.add_argument('--num-steps', type=int, default=30 * 1000)
    parser.add_argument('--num-particles', type=int, default=5)
    parser.add_argument('--log-every', type=int, default=3000)
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif args.device == 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)

    main(args)
