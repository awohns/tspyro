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
    scale_factor=None,
    num_eval_samples=500,
):
    assert isinstance(Model, type)

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    if device is None:
        device = torch.device("cpu")

    model = Model(
        ts=ts,
        leaf_location=leaf_location,
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
                internal_time = dist.LogNormal(model.prior_loc, model.prior_scale).mean.to(device=device)
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
    unbound_guide = guide

    if scale_factor is None:
        scale_factor = 1.0 / ts.num_nodes
    if scale_factor != 1.0:
        guide = poutine.scale(guide, scale_factor)
        model = poutine.scale(model, scale_factor)

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
    last_internal_log_time = unbound_guide.median()['internal_time'].log().clone()

    for step in range(steps):
        loss = svi.step() / ts.num_nodes if scale_factor == 1.0 else svi.step()
        if milestones is not None:
            optim.step()
        losses.append(loss)
        if step % log_every == 0 or step == steps - 1:
            loss = np.mean([svi.evaluate_loss() for _ in range(20)])
            with torch.no_grad():
                median = (
                    unbound_guide.median()
                )  # assess convergence of migration scale parameter
                try:
                    migration_scale = float(median["migration_scale"])
                    migration_scales.append(migration_scale)
                except KeyError:
                    migration_scale = None
            time_conv_diagnostic = torch.abs(last_internal_log_time - median['internal_time'].log()).mean().item()
            last_internal_log_time = median['internal_time'].log().clone()
            print(
                f"step {step} loss = {loss:0.5g}, "
                f"Migration scale = {migration_scale}, "
                f"time conv. diagnostic = {time_conv_diagnostic:0.5g}"
            )
            if 'internal_location' in median:
                loc = median['internal_location']
                delta = (loc.unsqueeze(-1) - loc.unsqueeze(-2)).pow(2.0).sum(-1).sqrt()
                diag = torch.arange(delta.size(-1))
                delta[diag, diag] = torch.inf
                print("delta_min", delta.min().item())

    final_elbo = np.mean([svi.evaluate_loss() for _ in range(num_eval_samples)])
    print("final_elbo: {:.4f}".format(final_elbo))

    median = unbound_guide.median()
    pyro_time, gaps, location, migration_scale = poutine.condition(model, median)()

    return pyro_time, location, migration_scale, unbound_guide, losses, final_elbo
