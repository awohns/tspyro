import os

import pyro
import pyslim
import pytest
import torch
import tsdate
from pyro import poutine
from tspyro.models import euclidean_migration
from tspyro.models import fit_guide
from tspyro.models import mean_field_location
from tspyro.models import NaiveModel
from tspyro.models import TimeDiffModel

REPO = os.path.dirname(os.path.dirname(__file__))
EXAMPLES = os.path.join(REPO, "examples")


@pytest.mark.parametrize(
    "Model, mask_locations",
    [
        (NaiveModel, 0),
        (NaiveModel, 10),
        (TimeDiffModel, 0),
    ],
)
def test_smoke(Model, mask_locations):
    ts = pyslim.load(os.path.join(EXAMPLES, "spatial_sim.trees")).simplify()
    recap_ts = ts.recapitate(recombination_rate=1e-8, Ne=50).simplify()

    lat_long = []
    for node in recap_ts.nodes():
        if node.individual != -1:
            ind = recap_ts.individual(node.individual)
            lat_long.append([ind.location[0], ind.location[1]])
    leaf_location = torch.tensor(
        lat_long[: recap_ts.num_samples], dtype=torch.get_default_dtype()
    )

    priors = tsdate.build_prior_grid(
        recap_ts,
        Ne=10000,
        approximate_priors=True,
        timepoints=100,
        progress=True,
    )

    # Optionally mask out the last mas_loctions-many nodes, to exclude them
    # from the migration likelihood.
    migration_likelihood = euclidean_migration
    if mask_locations:
        assert mask_locations < recap_ts.num_nodes
        mask = torch.ones(recap_ts.num_edges, dtype=torch.bool)
        for e, edge in enumerate(recap_ts.edges()):
            if edge.child >= mask_locations:
                mask[e] = False
        migration_likelihood = poutine.mask(migration_likelihood, mask=mask)

    pyro_time, location, migration_scale, guide, losses = fit_guide(
        recap_ts,
        leaf_location,
        priors,
        migration_likelihood=migration_likelihood,
        location_model=mean_field_location,
        steps=3,
        Model=Model,
    )

    # Test vectorized sampling.
    num_samples = 4
    with pyro.plate("particles", num_samples, dim=-2):
        guide_trace = poutine.trace(guide).get_trace()
        model_trace = poutine.trace(
            poutine.replay(guide.model, guide_trace)
        ).get_trace()
        print(list(model_trace.nodes))
