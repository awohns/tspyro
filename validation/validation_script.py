
import sys
sys.path.insert(0, "/Users/awohns/Documents/tspyro/")

import tspyro
import tskit
import tsdate

import numpy as np
import msprime
import torch
from pyro import poutine

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_squared_log_error


ts = tskit.load("slim_2d_continuous_recapitated_mutated.trees")

# Now let's randomly sample 200 leaf nodes
rng = np.random.default_rng(20)
random_sample = np.random.choice(np.arange(0, ts.num_samples), 200, replace=False)
sampled_ts = ts.simplify(samples=random_sample)


# Next, we need to get the locations of the nodes
node_locations = np.full((sampled_ts.num_nodes, 2), -1, dtype=float)
no_location_nodes = []
for node in sampled_ts.nodes():
    if node.individual != -1:
        node_locations[node.id, :] = sampled_ts.individual(node.individual).location[:2]
    else:
        no_location_nodes.append(node.id)
# We want a tensor with only the leaf nodes for inference
leaf_locations = torch.from_numpy(node_locations[sampled_ts.samples()])


# Let's perform joint inference of time and location

# We restrict inference of locations to nodes within 100 generations of the samples (which are all at time zero)
masked = sampled_ts.tables.nodes.time >= 100
mask = torch.ones(sampled_ts.num_edges, dtype=torch.bool)
for e, edge in enumerate(sampled_ts.edges()):
    if masked[edge.child]:
        mask[e] = False
migration_likelihood = tspyro.models.euclidean_migration
migration_likelihood = poutine.mask(
    migration_likelihood, mask=mask)

priors = tsdate.build_prior_grid(sampled_ts, Ne=1000)
slim_inferred_times_joint, slim_inferred_locations, slim_migration_scale_joint, slim_guide_joint, slim_losses_joint = tspyro.models.fit_guide(
    sampled_ts, leaf_location=leaf_locations,
    migration_likelihood=migration_likelihood,
    priors=priors, mutation_rate=1e-8, steps=10000, log_every=1000)



