import argparse
import pickle

import tspyro
import tskit
import tsdate

import numpy as np
import torch
from pyro import poutine

from fit import fit_guide
from run_val import load_data

from sklearn.metrics import mean_squared_error, mean_squared_log_error


def main(args):
    print(args)

    #ts = load_data(args)
    #priors = tsdate.build_prior_grid(ts, Ne=args.Ne)

    result = pickle.load(open(args.pkl, 'rb'))

    #dated = tsdate.date(ts, mutation_rate=1e-8, Ne=args.Ne)
    #tsdate_times = dated.tables.nodes.time

    inferred_times = result['inferred_times']
    true_times = result['true_times']

    pyro_time_msle = mean_squared_log_error(inferred_times, true_times)
    #tsdate_time_msle = mean_squared_log_error(tsdate_times, true_times)

    print("pyro_time_msle", pyro_time_msle)
    #print("pyro_time_msle", pyro_time_msle, "tsdate_time_msle", tsdate_time_msle)

    if 'inferred_internal_locations' in result:
        inferred_internal_locations = result['inferred_internal_locations']
        true_internal_locations = result['true_internal_locations']
        not_missing = ~np.isnan(true_internal_locations)[:, 0]
        rmse = np.sqrt(mean_squared_error(true_internal_locations[not_missing], inferred_internal_locations[not_missing]))
        print("pyro_spatial_rmse", rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze validation run')
    parser.add_argument('--pkl', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-nodes', type=int, default=500)
    parser.add_argument('--ts', type=str, default='slim_2d_continuous_recapitated_mutated.trees')
    parser.add_argument('--Ne', type=int, default=1000)
    args = parser.parse_args()

    main(args)
