import argparse
import os
import pickle

import numpy as np
import torch
import tsdate
import tskit
from tspyro import models


def create_filename(args, prefix):
    Ne = str(args.Ne)
    mutation_rate = str(args.mutation_rate)
    if args.euclidean_likelihood:
        likelihood = "euclidean"
    elif args.waypoint_likelihood:
        likelihood = "waypoint"
    if args.reparam_model:
        model = "reparam"
    elif args.meanfield_model:
        model = "meanfield"
    steps = str(args.steps)
    ts_fn = os.path.basename(args.tree_sequence)
    ts_fn = os.path.splitext(ts_fn)[0]
    filename = (
        args.output_path
        + ts_fn
        + "_"
        + prefix
        + "_"
        + Ne
        + "_"
        + mutation_rate
        + "_"
        + likelihood
        + "_"
        + model
        + "_"
        + steps
    )
    return filename


def main(args):
    """Main Entry Point"""
    ts = tskit.load(args.tree_sequence)
    # Simplify to modern samples
    # TODO: Add locations to ancient samples
    modern_samples = np.where(ts.tables.nodes.time == 0)[0]
    ts = ts.simplify(samples=modern_samples)
    ts = ts.simplify(np.arange(0, 10))
    ts = ts.keep_intervals([[0, 1e7]])
    # Get locations of samples from the tree sequence
    locations = []
    for indiv in ts.individuals():
        if len(indiv.location) > 0:
            # Bit of a hack, but the following is doubled as we
            # need to record a location for each chromosome
            locations.append(indiv.location)
            locations.append(indiv.location)
    locations = np.array(locations)
    leaf_location = torch.as_tensor(
        locations[: ts.num_samples, :], dtype=torch.get_default_dtype()
    )
    if args.euclidean_likelihood:
        likelihood = models.euclidean_migration
    elif args.waypoint_likelihood:
        raise NotImplementedError("Waypoint Model not implemented for world map yet")
    else:
        raise ValueError("Must specify a migration likelihood")
    if args.reparam_model:
        model = models.ReparamLocation(ts, leaf_location)
    elif args.meanfield_model:
        model = models.mean_field_location
    else:
        raise ValueError("Must specify a location model")

    # Create the priors for dates
    priors = tsdate.build_prior_grid(
        ts, Ne=args.Ne, approximate_priors=True, progress=args.progress
    )

    times, location, migration_scale, guide, losses = models.fit_guide(
        ts,
        leaf_location,
        priors,
        Ne=args.Ne,
        mutation_rate=args.mutation_rate,
        migration_likelihood=likelihood,
        location_model=model,
        steps=args.steps,
        log_every=args.log_every,
        Model=models.NaiveModel,
        migration_scale_init=args.migration_scale_init,
        learning_rate=args.learning_rate,
    )
    pickle.dump(times, open(create_filename(args, "times"), "wb"))
    pickle.dump(location, open(create_filename(args, "location"), "wb"))
    pickle.dump(migration_scale, open(create_filename(args, "migration_scale"), "wb"))
    # TODO: Why isn't this working?
    # torch.save(guide, create_filename(args, "guide"))
    # pickle.dump(guide, open(create_filename(args, "guide"), "wb"))
    pickle.dump(losses, open(create_filename(args, "losses"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs tspyro on a tree sequence")
    parser.add_argument(
        "tree_sequence",
        help="The path and name of the input tree sequence from which \
                        we estimate node ages.",
    )
    parser.add_argument(
        "output_path", help="The path to the directory where output will be saved"
    )
    parser.add_argument(
        "Ne", type=int, help="The effective population size to use in the prior"
    )
    parser.add_argument(
        "mutation_rate",
        type=float,
        help="The estimated mutation rate per unit of genome per generation.",
    )
    parser.add_argument(
        "--reparam-model",
        action="store_true",
        help="use the reparameterized location model",
    )
    parser.add_argument(
        "--meanfield-model",
        action="store_true",
        help="use the reparameterized location model",
    )
    parser.add_argument(
        "--euclidean-likelihood",
        action="store_true",
        help="use the euclidean migration likelihood model",
    )
    parser.add_argument(
        "--waypoint-likelihood",
        action="store_true",
        help="use the waypoint migration likelihood model",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps to run inference"
    )
    parser.add_argument(
        "--migration-scale-init",
        type=float,
        default=1,
        help="Initialization value for the migration scale",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate value to use in ELBO inference",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="How often do we log our inference progress?",
    )
    parser.add_argument(
        "-p", "--progress", action="store_true", help="Show progress bar."
    )
    args = parser.parse_args()
    main(args)
