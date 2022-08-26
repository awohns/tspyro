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
    dtype = torch.ones(1).dtype
    cg = CG(ts, time_cutoff=args.time_cutoff, strategy=args.strategy,
            device=device, dtype=dtype)
    cg.test_matmul(cg.b)
    cg.test_b_lambda_diag()

    cg.compute_heuristic_metrics()

    x_cholesky = cg.do_cholesky_inversion()
    x_cg = cg.do_cg(tol=1.0e-10)

    delta = (x_cg - x_cholesky).abs().max().item()
    print("delta between x CG and x cholesky: {:.2e}".format(delta))
    assert delta < 1.0e-8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')

    default_ts = 'slim_2d_continuous_recapitated_mutated.down_500_0.trees'

    parser.add_argument('--ts', type=str, default=default_ts)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--time-cutoff', type=float, default=250.0)
    parser.add_argument('--strategy', type=str, default='sever', choices=['sever', 'fill'])
    args = parser.parse_args()

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    elif args.device == 'cpu':
        torch.set_default_tensor_type(torch.DoubleTensor)

    main(args)
