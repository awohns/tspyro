import tskit
import numpy as np


def downsample_ts(filename, num_nodes, seed):
    ts = tskit.load(filename)
    np.random.seed(seed)
    random_sample = np.random.choice(np.arange(0, ts.num_samples), num_nodes, replace=False)
    sampled_ts = ts.simplify(samples=random_sample)
    sampled_ts.dump(filename.split('.')[0] + '.down_{}_{}.trees'.format(num_nodes, seed))
