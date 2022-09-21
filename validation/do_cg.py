import argparse
import pickle

import tskit

import torch
from cg import CG
import time


def load_data(args):
    t0 = time.time()
    ts = tskit.load(args.ts)
    t1 = time.time()
    print("Tree sequence {} has {} nodes. Took {:.2f} seconds to load.".format(args.ts, ts.num_samples, t1 - t0))
    return ts


def main(args):
    print(args)

    ts = load_data(args)

    device = torch.ones(1).device
    cg = CG(ts, time_cutoff=args.time_cutoff, strategy=args.strategy, device=device)
    cg.compute_heuristic_metrics()
    #cg.do_cg(tol=1.0e-9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')

    default_ts = 'slim_2d_continuous_recapitated_mutated.down_100_0.trees'
    default_ts = 'slim_cont_nocomp_N_2e4_Ne_8.5e3_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8000_ancs_79_rep_1.recap.trees'
    default_ts = 'slim_cont_nocomp_N_2e4_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8e4_ancs_790_rep_1.recap.trees'
    default_ts = 'slim_2d_cont_nocomp_N_4e3_sigma_0.5_mate_choice_0.5_max_dist_2_generations_8e3_ancients_70_rep_2.trees'
    default_ts = 'slim_2d_cont_nocomp_N_3e3_sigma_0.5_matechoice_0.5_maxdist_2_gens_8e3_len_2e8_ancs_7990_rep_2.mutated.recapitated.trees'
    default_ts = 'ancient_chr20.dates_added.simplified.dated.trees'
    #default_ts = 'ancient_chr20.dates_added.simplified.dated.down_500_0.trees'
    default_ts = 'ancient_chr20.dates_added.simplified.dated.down_50_0.trees'

    parser.add_argument('--ts', type=str, default=default_ts)
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'])
    parser.add_argument('--time-cutoff', type=float, default=200.0)
    parser.add_argument('--strategy', type=str, default='fill', choices=['sever', 'fill'])
    args = parser.parse_args()

    if args.device == 'gpu':
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    elif args.device == 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)

    main(args)
