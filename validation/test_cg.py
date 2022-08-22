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

    cg = CG(ts)
    print("num nodes", cg.num_nodes)


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
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif args.device == 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)

    main(args)
