import argparse
import pickle
from os.path import exists

import tsdate

import numpy as np

from run_val import load_data

from sklearn.metrics import mean_squared_error, mean_squared_log_error


def compute_baselines(ts_filename, Ne, true_internal_times, true_locations=None,
                      baselines_dir='./baselines/'):

    f = baselines_dir + 'baselines.{}.pkl'.format(ts_filename)
    if exists(f):
        return pickle.load(open(f, 'rb'))

    # else compute baseline metrics
    ts = load_data(ts_filename)

    tsdate_internal_times = tsdate.date(ts, mutation_rate=1e-8, Ne=Ne).tables.nodes.time[ts.num_samples:]
    tsdate_time_msle = mean_squared_log_error(tsdate_internal_times, true_internal_times)
    tsdate_time_male = np.mean(np.abs(np.log(tsdate_internal_times) - np.log(true_internal_times)))

    baselines = {'tsdate_time_msle': tsdate_time_msle,
                 'tsdate_time_male': tsdate_time_male}
    pickle.dump(baselines, open(f, 'wb'))

    return baselines


def main(args):
    result = pickle.load(open(args.pkl, 'rb'))

    inferred_times = result['inferred_times']
    inferred_internal_times = result['inferred_internal_times']
    true_times = result['true_times']
    true_internal_times = result['true_internal_times']
    ts_filename = result['ts_filename']
    Ne = result['Ne']

    print("final_elbo: {:.4f}".format(result['final_elbo']))

    baselines = compute_baselines(ts_filename, Ne, true_internal_times)
    for k, v in baselines.items():
        print(k + ': {:.4f}'.format(v))

    pyro_time_msle = mean_squared_log_error(inferred_internal_times, true_internal_times)
    pyro_time_male = np.mean(np.abs(np.log(inferred_internal_times) - np.log(true_internal_times)))

    print("pyro_time_msle: {:.4f}".format(pyro_time_msle))
    print("pyro_time_male: {:.4f}".format(pyro_time_male))

    if 'inferred_internal_locations' in result:
        inferred_internal_locations = result['inferred_internal_locations']
        true_internal_locations = result['true_internal_locations']
        not_missing = ~np.isnan(true_internal_locations)[:, 0]
        rmse = np.sqrt(mean_squared_error(true_internal_locations[not_missing], inferred_internal_locations[not_missing]))
        mae = np.power(true_internal_locations[not_missing] - inferred_internal_locations[not_missing], 2).sum(-1)
        mae = np.sqrt(mae).mean()
        print("pyro_spatial_rmse: {:.4f}".format(rmse))
        print("pyro_spatial_mae: {:.4f}".format(mae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze validation run')
    parser.add_argument('--pkl', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-nodes', type=int, default=500)
    parser.add_argument('--ts', type=str, default='slim_2d_continuous_recapitated_mutated.trees')
    parser.add_argument('--Ne', type=int, default=1000)
    args = parser.parse_args()

    main(args)
