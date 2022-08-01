"""
Functions to perform inference on a tree sequence
"""
import tsdate

from . import models


def infer_geotime(
    ts, leaf_location, mutation_rate, priors=None, Ne=10000, steps=1001
):
    """
    Takes a tree sequence and infers the time, and optionally the location, of its ancestral haplotypes.
    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`, treated as
        one whose non-sample nodes are undated and locations are unknown.
    :param float mutation_rate: The estimated mutation rate per unit of genome per
        unit time. The dating algorithm will use a mutation rate clock to
        help estimate node dates. 
    :param float Ne: The estimated (diploid) effective population size used to construct
        the (default) conditional coalescent prior. This is what is used when ``priors``
        is ``None``: a positive ``Ne`` value is therefore required in this case.
        Conversely, if ``priors`` is not ``None``, no ``Ne`` value should be given.
    :return: The inferred times and locations of the ancestral haplotypes in the
        given tree sequence, the inferred migration scale, the guide, and the losses
        at each recorded step in the inference.
    """
    if priors is None:
        priors = tsdate.build_prior_grid(ts, Ne=Ne)
    return models.fit_guide(ts, leaf_location, priors=priors, Ne=Ne, mutation_rate=mutation_rate, steps=steps)
