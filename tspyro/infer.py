import numpy as np
import pyro
import torch
from pyro import poutine
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide import init_to_median

from . import geography
from . import model


def infer_pyro(
    ts, priors, leaf_location=None, Ne=10000, mutation_rate=1e-8, steps=1001
):

    pyro.set_rng_seed(20210518)
    pyro.clear_param_store()

    naive_model = model.NaiveModel(
        ts,
        leaf_location=leaf_location,
        prior=priors,
        Ne=Ne,
        mutation_rate=mutation_rate,
    )

    # autoguide initialises location param, but we want a different initialiser for
    # each pyro sample statement if we wanted to initialise global params, then check
    # 'if site["name"] == "global_param_name":' (this is a property of the approx
    # posterior, not model, hence it's here and not in forward())
    def init_loc_fn(site):
        if site["name"] == "internal_time":
            prior_init = np.einsum(
                "t,nt->n", priors.timepoints, (priors.grid_data[:])
            ) / np.sum(priors.grid_data[:], axis=1)
            internal_time = torch.as_tensor(prior_init, dtype=torch.float)  # / Ne
            return internal_time.clamp(min=0.1)
        # GEOGRAPHY.
        if site["name"] == "internal_location":
            initial_guess_loc = geography.get_ancestral_geography(
                ts, leaf_location
            )  # np.repeat([[10,10]], sample_ts.num_samples, axis=0))
            return initial_guess_loc
        return init_to_median(
            site
        )  # automatic strategy (the default) if not "internal_time"

    # guide = AutoDelta(model) # Simple point estimate
    guide = AutoNormal(
        naive_model, init_scale=0.01, init_loc_fn=init_loc_fn
    )  # Mean field (fully Bayesian)
    # guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model, init_scale=0.01)
    # , init_loc_fn=init_loc_fn) # bigger matricies means clipped adam works better
    # optim = Adam({"lr": 0.05})
    optim = pyro.optim.ClippedAdam(
        {"lr": 0.005, "lrd": 0.1 ** (1 / max(1, steps))}
    )  # lrd (learning rate decay) decreases lr over num of steps from lr to lr * 0.01
    # optim = pyro.optim.ClippedAdam({"lr": 0.005})
    svi = SVI(naive_model, guide, optim, Trace_ELBO())
    guide()  # initialises the guide
    losses = []
    migration_scales = []
    for step in range(steps):
        loss = svi.step() / ts.num_nodes
        losses.append(loss)
        if step % 100 == 0:
            with torch.no_grad():
                median = (
                    guide.median()
                )  # assess convergence of migration scale parameter
                migration_scale = float(median["migration_scale"])
                migration_scales.append(migration_scale)
            print(
                f"step {step} loss = {loss:0.5g},"
                "Migration scale= {migration_scale:0.3g}"
            )
    median = guide.median()
    pyro_time, gaps, location, migration_scale = poutine.condition(
        naive_model, median
    )()
    return naive_model, pyro_time, gaps, location, migration_scale, guide
