"""
Functions to perform inference on a tree sequence
"""
import tsdate

from . import uniform_geography


def infer_geotime(
    ts, leaf_location, priors=None, Ne=10000, mutation_rate=1e-8, steps=1001
):
    if priors is None:
        priors = tsdate.build_prior_grid(ts, Ne=10000)
    return uniform_geography.guide(ts, leaf_location, priors, Ne, mutation_rate, steps)
