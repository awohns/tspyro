import numpy as np
import pyro
import torch
from pyro import poutine
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import MultiStepLR, ClippedAdam
from tspyro.ops import get_ancestral_geography

from models import NaiveModel, mean_field_location


def fit_guide(
    ts,
    leaf_location,
    priors,
    init_times=None,
    init_loc=None,
    Ne=10000,
    mutation_rate=1e-8,
    migration_scale_init=1,
    milestones=None,
    gamma=0.2,
    steps=1001,
    Model=NaiveModel,
    migration_likelihood=None,
    location_model=mean_field_location,
    learning_rate=0.005,
    learning_rate_decay=0.1,
    log_every=100,
    clip_norm=10.0,
    device=None,
    seed=0,
):
    assert isinstance(Model, type)

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    if device is None:
        device = torch.device("cpu")

    model = Model(
        ts=ts,
        leaf_location=leaf_location,
        prior=priors,
        Ne=Ne,
        mutation_rate=mutation_rate,
        migration_likelihood=migration_likelihood,
        location_model=location_model,
    )
    model = model.to(device=device)

    def init_loc_fn(site):
        # TIME
        if site["name"] == "internal_time":
            if init_times is not None:
                internal_time = init_times
            else:
                prior_grid_data = priors.grid_data[
                    priors.row_lookup[ts.num_samples :]  # noqa: E203
                ]
                prior_init = np.einsum(
                    "t,nt->n", priors.timepoints, (prior_grid_data)
                ) / np.sum(prior_grid_data, axis=1)
                internal_time = torch.as_tensor(
                    prior_init, dtype=torch.get_default_dtype(), device=device
                )  # / Ne
                internal_time = internal_time.nan_to_num(10)
            return internal_time.clamp(min=0.1)
        if site["name"] == "internal_diff":
            internal_diff = model.prior_diff_loc.exp()
            internal_diff.sub_(1).clamp_(min=0.1)
            return internal_diff
        # GEOGRAPHY.
        if site["name"] == "internal_location":
            if init_loc is not None:
                initial_guess_loc = init_loc
            else:
                initial_guess_loc = get_ancestral_geography(ts, leaf_location)
            return initial_guess_loc
        if site["name"] == "internal_delta":
            return torch.zeros(site["fn"].shape())
        if site["name"] == "migration_scale":
            return torch.tensor(float(migration_scale_init), device=device)
        raise NotImplementedError("Missing init for {}".format(site["name"]))

    guide = AutoNormal(
        model, init_scale=0.01, init_loc_fn=init_loc_fn
    )  # Mean field (fully Bayesian)

    if milestones is None:
        optim = ClippedAdam(
            {"lr": learning_rate, "lrd": learning_rate_decay ** (1 / max(1, steps)), "clip_norm": clip_norm}
        )
    else:
        optim = MultiStepLR({'optimizer': torch.optim.Adam,
                             'optim_args': {'lr': learning_rate},
                             'gamma': 0.2,
                             'milestones': milestones})

    svi = SVI(model, guide, optim, Trace_ELBO())
    guide()  # initialises the guide

    losses = []
    migration_scales = []

    for step in range(steps):
        loss = svi.step() / ts.num_nodes
        losses.append(loss)
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                median = (
                    guide.median()
                )  # assess convergence of migration scale parameter
                try:
                    migration_scale = float(median["migration_scale"])
                    migration_scales.append(migration_scale)
                except KeyError:
                    migration_scale = None
            print(
                f"step {step} loss = {loss:0.5g}, "
                f"Migration scale= {migration_scale}"
            )

    median = guide.median()
    pyro_time, gaps, location, migration_scale = poutine.condition(model, median)()

    return pyro_time, location, migration_scale, guide, losses
