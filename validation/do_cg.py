import argparse
import pickle

import tskit

import numpy as np
import torch
from cg import CG


def load_data(args):
    ts = tskit.load(args.ts).simplify()
    print("Tree sequence {} has {} nodes.".format(args.ts, ts.num_samples))
    return ts


def main(args):
    print(args)

    ts = load_data(args)

    device = torch.ones(1).device
    cg = CG(ts, time_cutoff=args.time_cutoff, strategy=args.strategy, device=device)
    cg.compute_heuristic_metrics()
    cg.do_cg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')

    default_ts = 'slim_2d_continuous_recapitated_mutated.down_100_0.trees'
    default_ts = 'slim_cont_nocomp_N_2e4_Ne_8.5e3_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8000_ancs_79_rep_1.recap.trees'
    default_ts = 'slim_cont_nocomp_N_2e4_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8e4_ancs_790_rep_1.recap.trees'
    #default_ts = 'slim_cont_nocomp_N_2e4_Ne_1e4_mu_1e-8_rec_1e-8_sig_0.5_mate_0.5_maxdist_2_gens_8000_ancs_70_rep_1.recap.trees'

    parser.add_argument('--ts', type=str, default=default_ts)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--time-cutoff', type=float, default=100.0)
    parser.add_argument('--strategy', type=str, default='sever', choices=['sever', 'fill'])
    args = parser.parse_args()

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif args.device == 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)

    main(args)
