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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tspyro validation')

    default_ts = 'slim_2d_continuous_recapitated_mutated.down_100_0.trees'

    parser.add_argument('--ts', type=str, default=default_ts)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif args.device == 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)

    main(args)
