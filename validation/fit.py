import time
import numpy as np
import pyro
import torch
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal, AutoDelta, AutoLowRankMultivariateNormal
from pyro.optim import MultiStepLR, ClippedAdam
from tspyro.ops import get_ancestral_geography

from models import NaiveModel, mean_field_location


def fit_guide(
    tree_seq,
    leaf_location,
    init_times=None,
    init_loc=None,
    Ne=1000,
    mutation_rate=1e-8,
    migration_scale_init=1,
    milestones=None,
    gamma=0.2,
    inference='svi',
    steps=1001,
    Model=NaiveModel,
    migration_likelihood=None,
    location_model=mean_field_location,
    learning_rate=0.005,
    learning_rate_decay=0.1,
    log_every=100,
    clip_norm=100.0,
    device=torch.device("cpu"),
    seed=0,
    scale_factor=None,
    num_eval_samples=500,
    gap_prefactor=1.0,
    gap_exponent=1.0,
    min_gap=1.0,
):
    assert isinstance(Model, type)

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    model = Model(
        ts=tree_seq,
        leaf_location=leaf_location,
        Ne=Ne,
        mutation_rate=mutation_rate,
        migration_likelihood=migration_likelihood,
        location_model=location_model,
        gap_prefactor=gap_prefactor,
        gap_exponent=gap_exponent,
        min_gap=min_gap,
    )

    prior_loc = model.prior_loc
    prior_scale = model.prior_scale
    prior_diff_loc = model.prior_diff_loc if hasattr(model, 'prior_diff_loc') else None
    model = model.to(device=device)

    def init_loc_fn(site):
        # TIME
        if site["name"] == "internal_time":
            if init_times is not None:
                internal_time = init_times
            else:
                internal_time = dist.LogNormal(prior_loc, prior_scale).mean.to(device=device)
            return internal_time.clamp(min=0.1)
        if site["name"] == "internal_diff":
            internal_diff = prior_diff_loc.exp()
            internal_diff.sub_(1).clamp_(min=0.1)
            return internal_diff
        # GEOGRAPHY.
        if site["name"] == "internal_location":
            if init_loc is not None:
                initial_guess_loc = init_loc
            else:
                initial_guess_loc = get_ancestral_geography(tree_seq, leaf_location.data.cpu().numpy()).to(device=device)
            return initial_guess_loc
        if site["name"] == "internal_delta":
            return torch.zeros(site["fn"].shape())
        if site["name"] == "migration_scale":
            return torch.tensor(float(migration_scale_init), device=device)
        raise NotImplementedError("Missing init for {}".format(site["name"]))

    if inference == 'svi':
        guide = AutoNormal(
            model, init_scale=1.0e-2, init_loc_fn=init_loc_fn
        )  # Mean field (fully Bayesian)
    elif inference == 'svilowrank':
        guide = AutoLowRankMultivariateNormal(
            model, init_scale=1.0e-2, init_loc_fn=init_loc_fn, rank=200
        )  # Mean field (fully Bayesian)
    elif inference == 'map':
        guide = AutoDelta(
            model, init_loc_fn=init_loc_fn)
    unbound_guide = guide

    if scale_factor is None:
        scale_factor = 1.0 / tree_seq.num_nodes
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
    ts = [time.time()]
    migration_scales = []
    if Model.__name__ == 'NaiveModel':
        key_name = 'internal_time'
        last_internal_log_time = unbound_guide.median()[key_name].log().clone()
    elif Model.__name__ == 'TimeDiffModel':
        key_name = 'internal_diff'
        last_internal_log_time = unbound_guide.median()[key_name].log().clone()
    elif Model.__name__ == 'ConditionedTimesNaiveModel':
        key_name = 'internal_location'
        last_internal_log_time = unbound_guide.median()[key_name].clone()

    for step in range(steps):
        loss = svi.step() / tree_seq.num_nodes if scale_factor == 1.0 else svi.step()
        if milestones is not None:
            optim.step()
        ts.append(time.time())
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
                    try:
                        migration_scale = pyro.param("migration_scale").item()
                    except:
                        migration_scale = None
            if 'location' not in key_name:
                conv_diagnostic = torch.abs(last_internal_log_time - median[key_name].log()).mean().item()
                last_internal_log_time = median[key_name].log().clone()
            else:
                conv_diagnostic = torch.abs(last_internal_log_time - median[key_name]).mean().item()
                last_internal_log_time = median[key_name].clone()
            ips = 0.0 if step == 0 or steps <= log_every else log_every / (ts[-1] - ts[-1 - log_every])
            print(
                f"step {step} loss = {loss:0.5g}, "
                f"Migration scale = {migration_scale}, "
                f"conv. diagnostic = {conv_diagnostic:0.3g}, "
                f"iter. per sec. = {ips:0.2f}"
            )

    num_eval_samples = num_eval_samples if steps > 10000 else 1
    final_elbo = np.mean([svi.evaluate_loss() for _ in range(num_eval_samples)])
    print("final_elbo: {:.4f}".format(final_elbo))

    median = unbound_guide.median()
    pyro_time, gaps, location, migration_scale = poutine.condition(model, median)()

    return pyro_time, location, migration_scale, unbound_guide, losses, final_elbo
