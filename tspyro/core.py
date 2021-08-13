"""
Functions to perform inference on a tree sequence
"""
from . import infer


def infer_geotime(
    ts, priors, leaf_location=None, Ne=10000, mutation_rate=1e-8, steps=1001
):
    return infer.date_pyro(ts, priors, leaf_location, Ne, mutation_rate, steps)
