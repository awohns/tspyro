import argparse
import pickle
from os.path import exists

import tsdate

import numpy as np

from run_val import load_data

from sklearn.metrics import mean_squared_error, mean_squared_log_error
from scipy.stats import pearsonr, spearmanr


def compute_time_metrics(true_internal_times, inferred_internal_times):
    result = {}
    result['time_msle'] = mean_squared_log_error(inferred_internal_times, true_internal_times)
    result['time_male'] = np.mean(np.abs(np.log(inferred_internal_times) - np.log(true_internal_times)))
    result['time_log_bias'] = np.mean(np.log(inferred_internal_times) - np.log(true_internal_times))
    result['time_pearson'] = pearsonr(inferred_internal_times, true_internal_times)[0]
    result['time_spearman'] = spearmanr(inferred_internal_times, true_internal_times)[0]

    median_time = np.median(true_internal_times)
    early, late = true_internal_times <= median_time, true_internal_times >= median_time
    result['time_msle_late'] = mean_squared_log_error(inferred_internal_times[late], true_internal_times[late])
    result['time_msle_early'] = mean_squared_log_error(inferred_internal_times[early], true_internal_times[early])
    result['time_male_late'] = np.mean(np.abs(np.log(inferred_internal_times[late]) - np.log(true_internal_times[late])))
    result['time_male_early'] = np.mean(np.abs(np.log(inferred_internal_times[early]) - np.log(true_internal_times[early])))
    result['time_log_bias_late'] = np.mean(np.log(inferred_internal_times[late]) - np.log(true_internal_times[late]))
    result['time_log_bias_early'] = np.mean(np.log(inferred_internal_times[early]) - np.log(true_internal_times[early]))
    return result


def compute_spatial_metrics(true_internal_locs, inferred_internal_locs, true_internal_times):
    result = {}
    not_missing = ~np.isnan(true_internal_locs)[:, 0]
    rmse = np.sqrt(mean_squared_error(true_internal_locs[not_missing], inferred_internal_locs[not_missing]))
    mae = np.sqrt(np.power(true_internal_locs[not_missing] - inferred_internal_locs[not_missing], 2).sum(-1)).mean()
    result['spatial_rmse'] = rmse
    result['spatial_mae'] = mae

    median_time = np.median(true_internal_times)
    early, late = true_internal_times <= median_time, true_internal_times >= median_time
    early, late = early & not_missing, late & not_missing
    rmse_late = np.sqrt(mean_squared_error(true_internal_locs[late], inferred_internal_locs[late]))
    rmse_early = np.sqrt(mean_squared_error(true_internal_locs[early], inferred_internal_locs[early]))
    mae_late = np.sqrt(np.power(true_internal_locs[late] - inferred_internal_locs[late], 2).sum(-1)).mean()
    mae_early = np.sqrt(np.power(true_internal_locs[early] - inferred_internal_locs[early], 2).sum(-1)).mean()
    result['spatial_rmse_late'] = rmse_late
    result['spatial_rmse_early'] = rmse_early
    result['spatial_mae_late'] = mae_late
    result['spatial_mae_early'] = mae_early
    return result


def compute_baselines(ts_filename, Ne, true_internal_times, true_locations=None,
                      baselines_dir='./baselines/'):

    f = baselines_dir + 'baselines.{}.pkl'.format(ts_filename)
    if exists(f):
        return pickle.load(open(f, 'rb'))

    # else compute baseline metrics
    ts = load_data(ts_filename)

    tsdate_internal_times = tsdate.date(ts, mutation_rate=1e-8, Ne=Ne).tables.nodes.time[ts.num_samples:]
    time_metrics = compute_time_metrics(true_internal_times, tsdate_internal_times)

    pickle.dump(time_metrics, open(f, 'wb'))

    return time_metrics


def main(args):
    result = pickle.load(open(args.pkl, 'rb'))

    inferred_internal_times = result['inferred_internal_times']
    true_internal_times = result['true_internal_times']
    ts_filename = result['ts_filename']
    Ne = result['Ne']

    print("final_elbo: {:.4f}".format(result['final_elbo']))

    baselines = compute_baselines(ts_filename, Ne, true_internal_times)
    for k, v in baselines.items():
        print('[tsdate] ' + k + ': {:.4f}'.format(v))

    pyro_metrics = {}
    pyro_metrics.update(compute_time_metrics(true_internal_times, inferred_internal_times))

    if 'inferred_internal_locations' in result:
        inferred_internal_locs = result['inferred_internal_locations']
        true_internal_locs = result['true_internal_locations']
        pyro_metrics.update(compute_spatial_metrics(true_internal_locs, inferred_internal_locs, true_internal_times))

    for k, v in pyro_metrics.items():
        print('[pyro-mig-{}] '.format(result['migration']) + k + ': {:.4f}'.format(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze validation run')
    parser.add_argument('--pkl', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-nodes', type=int, default=500)
    parser.add_argument('--ts', type=str, default='slim_2d_continuous_recapitated_mutated.trees')
    parser.add_argument('--Ne', type=int, default=1000)
    args = parser.parse_args()

    main(args)
