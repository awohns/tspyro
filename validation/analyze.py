import json
import math
import argparse
import pickle
from os.path import exists

import tsdate
import tskit

import numpy as np

from sklearn.metrics import mean_squared_error, mean_squared_log_error
from scipy.stats import pearsonr, spearmanr
from tspyro.ops import get_ancestral_geography
from util import get_metadata


def load_data(filename):
    ts = tskit.load(filename).simplify()
    print("Tree sequence has {} nodes".format(ts.num_samples))
    return ts


def compute_time_metrics(true_internal_times, inferred_internal_times):
    result = {}
    result['time_rmsle'] = np.sqrt(mean_squared_log_error(inferred_internal_times, true_internal_times))
    result['time_male'] = np.mean(np.abs(np.log(inferred_internal_times) - np.log(true_internal_times)))
    result['time_log_bias'] = np.mean(np.log(inferred_internal_times) - np.log(true_internal_times))
    result['time_pearson'] = pearsonr(inferred_internal_times, true_internal_times)[0]
    result['time_spearman'] = spearmanr(inferred_internal_times, true_internal_times)[0]

    median_time = np.median(true_internal_times)
    recent, ancient = true_internal_times <= median_time, true_internal_times > median_time
    result['time_rmsle_ancient'] = np.sqrt(mean_squared_log_error(inferred_internal_times[ancient], true_internal_times[ancient]))
    result['time_rmsle_recent'] = np.sqrt(mean_squared_log_error(inferred_internal_times[recent], true_internal_times[recent]))
    result['time_male_ancient'] = np.mean(np.abs(np.log(inferred_internal_times[ancient]) - np.log(true_internal_times[ancient])))
    result['time_male_recent'] = np.mean(np.abs(np.log(inferred_internal_times[recent]) - np.log(true_internal_times[recent])))
    result['time_log_bias_ancient'] = np.mean(np.log(inferred_internal_times[ancient]) - np.log(true_internal_times[ancient]))
    result['time_log_bias_recent'] = np.mean(np.log(inferred_internal_times[recent]) - np.log(true_internal_times[recent]))
    return result


def compute_spatial_metrics(true_internal_locs, inferred_internal_locs, true_internal_times):
    result = {}
    not_missing = ~np.isnan(true_internal_locs)[:, 0]
    rmse = np.sqrt(mean_squared_error(true_internal_locs[not_missing], inferred_internal_locs[not_missing]))
    mae = np.sqrt(np.power(true_internal_locs[not_missing] - inferred_internal_locs[not_missing], 2).sum(-1)).mean()
    result['spatial_rmse'] = rmse
    result['spatial_mae'] = mae

    median_time = np.median(true_internal_times)
    recent, ancient = true_internal_times <= median_time, true_internal_times > median_time
    recent, ancient = recent & not_missing, ancient & not_missing
    rmse_ancient = np.sqrt(mean_squared_error(true_internal_locs[ancient], inferred_internal_locs[ancient]))
    rmse_recent = np.sqrt(mean_squared_error(true_internal_locs[recent], inferred_internal_locs[recent]))
    mae_ancient = np.sqrt(np.power(true_internal_locs[ancient] - inferred_internal_locs[ancient], 2).sum(-1)).mean()
    mae_recent = np.sqrt(np.power(true_internal_locs[recent] - inferred_internal_locs[recent], 2).sum(-1)).mean()
    result['spatial_rmse_ancient'] = rmse_ancient
    result['spatial_rmse_recent'] = rmse_recent
    result['spatial_mae_ancient'] = mae_ancient
    result['spatial_mae_recent'] = mae_recent
    return result


def compute_baselines(ts_filename, Ne=None, mu=1.0e-8, baselines_dir='./baselines/'):
    ts = load_data(ts_filename)
    locations, true_times, is_leaf, is_internal = get_metadata(ts)

    f = baselines_dir + 'baselines.{}.pkl'.format(ts_filename)
    if exists(f):
        metrics = pickle.load(open(f, 'rb'))
    else:  # else compute baseline metrics
        #dated_ts = tsdate.date(ts, mutation_rate=mu, Ne=Ne)
        #tsdate_times = dated_ts.tables.nodes.time
        tsdate_times = ts.tables.nodes.time

        #unconstrained_inferred_times = np.zeros(dated_ts.num_nodes)
        #for node in dated_ts.nodes():
        #    if node.id not in dated_ts.samples():
        #        unconstrained_inferred_times[node.id] = json.loads(node.metadata)["mn"]
        #tsdate_times = unconstrained_inferred_times

        tsdate_internal_times = tsdate_times[is_internal]
        metrics = {'tsdate_times': tsdate_times}
        #metrics.update(compute_time_metrics(true_times[is_internal], tsdate_internal_times))

        has_locations = np.isnan(locations).sum().item() < locations.size
        if has_locations:
            ancestral_locs = get_ancestral_geography(ts, locations[is_leaf]).data.cpu().numpy()
            metrics.update(compute_spatial_metrics(locations[is_internal], ancestral_locs, true_times[is_internal]))

        pickle.dump(metrics, open(f, 'wb'))

    return metrics, locations, true_times, is_leaf, is_internal


def main(args):
    print("\n" + args.pkl)
    result = pickle.load(open(args.pkl, 'rb'))

    inferred_internal_times = result['inferred_internal_times']
    ts_filename = result['ts_filename']

    Ne = result['Ne']
    mu = result['mu']

    print("final_elbo: {:.4f}".format(result['final_elbo']))

    baselines, locations, true_times, is_leaf, is_internal = compute_baselines(ts_filename, Ne=Ne, mu=mu)
    if False:
        inferred_times = np.zeros(true_times.shape)
        inferred_times[is_internal] = inferred_internal_times
        inferred_internal_times2 = inferred_internal_times.copy()
        inferred_internal_times = tsdate.core.constrain_ages_topo(ts, inferred_times, 0.1)[is_internal]
        delta = np.abs(inferred_internal_times - inferred_internal_times2).max()
        #print("conc delta", delta)

    for k, v in baselines.items():
        if v.size == 1:
            print('[tsdate/anc] ' + k + ': {:.5f}'.format(v))

    pyro_metrics = {}
    pyro_metrics['inferred_internal_times'] = inferred_internal_times
    #pyro_metrics.update(compute_time_metrics(true_times[is_internal], inferred_internal_times))

    if 'inferred_internal_locations' in result:
        inferred_internal_locs = result['inferred_internal_locations']
        true_internal_locs = locations[is_internal]
        pyro_metrics.update(compute_spatial_metrics(true_internal_locs, inferred_internal_locs, true_times[is_internal]))

    for k, v in pyro_metrics.items():
        if v.size == 1:
            print('[pyro-mig-{}] '.format(result['migration']) + k + ': {:.5f}'.format(v))

    for k, v in baselines.items():
        if v.size == 1:
            pyro_metrics['tsdate_' + k] = v

    pyro_metrics['tsdate_internal_times'] = baselines['tsdate_times'][is_internal]
    pyro_metrics['true_internal_times'] = true_times[is_internal]

    f = 'metrics/metrics.{}'.format(args.pkl.split('/')[-1])
    pickle.dump(pyro_metrics, open(f, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze validation run')
    parser.add_argument('--pkl', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time-cutoff', type=float, default=100.0)
    parser.add_argument('--num-nodes', type=int, default=500)
    args = parser.parse_args()

    main(args)
