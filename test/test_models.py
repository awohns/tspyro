import os

import pyslim
import torch
import tsdate
from tspyro.models import euclidean_migration
from tspyro.models import fit_guide
from tspyro.models import mean_field_location
from tspyro.models import NaiveModel

REPO = os.path.dirname(os.path.dirname(__file__))
EXAMPLES = os.path.join(REPO, "examples")


def test_smoke():
    ts = pyslim.load(os.path.join(EXAMPLES, "two_islands.trees")).simplify()
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

    fit_guide(
        recap_ts,
        leaf_location,
        priors,
        migration_likelihood=euclidean_migration,
        location_model=mean_field_location,
        steps=10,
        Model=NaiveModel,
    )
