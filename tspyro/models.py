import itertools
import operator

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide import init_to_median
from pyro.nn import PyroModule
from tspyro.diffusion import ApproximateMatrixExponential
from tspyro.diffusion import WaypointDiffusion2D


class BaseModel(PyroModule):
    def __init__(
        self, ts, *, Ne, prior, leaf_location=None, mutation_rate=1e-8, penalty=100.0
    ):
        super().__init__()
        nodes = ts.tables.nodes
        edges = ts.tables.edges
        self.is_leaf = torch.tensor((nodes.flags & 1).astype(bool), dtype=torch.bool)
        self.is_internal = ~self.is_leaf
        self.num_nodes = len(self.is_leaf)
        self.num_internal = self.is_internal.sum().item()

        self.ts = ts

        self.parent = torch.tensor(edges.parent, dtype=torch.long)
        self.child = torch.tensor(edges.child, dtype=torch.long)
        self.span = torch.tensor(edges.right - edges.left, dtype=torch.float)
        self.mutations = torch.tensor(
            self.get_mut_edges(), dtype=torch.float
        )  # this is an int, but we optimise with float for pytorch

        self.penalty = float(penalty)
        self.Ne = float(Ne)
        self.mutation_rate = mutation_rate

        # conditional coalescent prior
        timepoints = torch.as_tensor(prior.timepoints, dtype=torch.float)
        timepoints = timepoints.mul(2 * Ne).log1p()
        grid_data = torch.as_tensor(prior.grid_data[:], dtype=torch.float)
        grid_data = grid_data / grid_data.sum(1, True)
        self.prior_loc = torch.einsum("t,nt->n", timepoints, grid_data)
        deltas = (timepoints - self.prior_loc[:, None]) ** 2
        self.prior_scale = torch.einsum("nt,nt->n", deltas, grid_data).sqrt()
        self.leaf_location = leaf_location

    def get_mut_edges(self):
        """
        Get the number of mutations on each edge in the tree sequence.
        """
        ts = self.ts
        edge_diff_iter = ts.edge_diffs()
        right = 0
        edges_by_child = {}  # contains {child_node:edge_id}
        mut_edges = np.zeros(ts.num_edges, dtype=np.int64)
        for site in ts.sites():
            while right <= site.position:
                (left, right), edges_out, edges_in = next(edge_diff_iter)
                for e in edges_out:
                    del edges_by_child[e.child]
                for e in edges_in:
                    assert e.child not in edges_by_child
                    edges_by_child[e.child] = e.id
            for m in site.mutations:
                # In some cases, mutations occur above the root
                # These don't provide any information for the inside step
                if m.node in edges_by_child:
                    edge_id = edges_by_child[m.node]
                    mut_edges[edge_id] += 1
        return mut_edges

    def weighted_geographic_center(self, lat_list, long_list, weights):
        x = list()
        y = list()
        z = list()
        if len(lat_list) == 1 and len(long_list) == 1:
            return (lat_list[0], long_list[0])
        lat_radians = np.radians(lat_list)
        long_radians = np.radians(long_list)
        x = np.cos(lat_radians) * np.cos(long_radians)
        y = np.cos(lat_radians) * np.sin(long_radians)
        z = np.sin(lat_radians)
        weights = np.array(weights)
        central_latitude, central_longitude = self.radians_center_weighted(
            x, y, z, weights
        )
        return (np.degrees(central_latitude), np.degrees(central_longitude))

    def edges_by_parent_asc(self, ts):
        """
        Return an itertools.groupby object of edges grouped by parent in ascending order
        of the time of the parent. Since tree sequence properties guarantee that edges
        are listed in nondecreasing order of parent time
        (https://tskit.readthedocs.io/en/latest/data-model.html#edge-requirements)
        we can simply use the standard edge order
        """
        return itertools.groupby(ts.edges(), operator.attrgetter("parent"))

    def edge_span(self, edge):
        return edge.right - edge.left

    def average_edges(self, parent_edges, locations):
        parent = parent_edges[0]
        edges = parent_edges[1]

        child_spanfracs = list()
        child_lats = list()
        child_longs = list()

        for edge in edges:
            child_spanfracs.append(self.edge_span(edge))
            child_lats.append(locations[edge.child][0])
            child_longs.append(locations[edge.child][1])
        val = self.weighted_geographic_center(
            child_lats, child_longs, np.ones_like(len(child_lats))
        )
        return parent, val

    def radians_center_weighted(self, x, y, z, weights):
        total_weight = np.sum(weights)
        weighted_avg_x = np.sum(weights * np.array(x)) / total_weight
        weighted_avg_y = np.sum(weights * np.array(y)) / total_weight
        weighted_avg_z = np.sum(weights * np.array(z)) / total_weight
        central_longitude = np.arctan2(weighted_avg_y, weighted_avg_x)
        central_square_root = np.sqrt(
            weighted_avg_x * weighted_avg_x + weighted_avg_y * weighted_avg_y
        )
        central_latitude = np.arctan2(weighted_avg_z, central_square_root)
        return central_latitude, central_longitude

    def get_ancestral_geography(self, ts, sample_locations, show_progress=False):
        """
        Use dynamic programming to find approximate posterior to sample from
        """
        locations = np.zeros((ts.num_nodes, 2))
        locations[ts.samples()] = sample_locations
        fixed_nodes = set(ts.samples())

        # Iterate through the nodes via groupby on parent node
        for parent_edges in self.edges_by_parent_asc(ts):
            if parent_edges[0] not in fixed_nodes:
                parent, val = self.average_edges(parent_edges, locations)
                locations[parent] = val
        return torch.tensor(locations[ts.num_samples :])  # noqa


class NaiveModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.migration_likelihood = kwargs.pop("migration_likelihood", None)
        super.__init__(*args, **kwargs)

    def forward(self):
        # First sample times from an improper uniform distribution which we denote
        # via .mask(False). Note only the internal nodes are sampled; leaves are
        # fixed at zero.
        # Note this isn't a coalescent prior, but some are available at:
        # https://docs.pyro.ai/en/stable/_modules/pyro/distributions/coalescent.html
        with pyro.plate("internal_nodes", self.num_internal):
            internal_time = pyro.sample(
                "internal_time",
                # dist.Exponential(1).mask(False),
                dist.LogNormal(self.prior_loc, self.prior_scale).mask(
                    True
                ),  # False turns off prior but uses it for initialisation
            )  # internal time is modelled in logspace

            # GEOGRAPHY.
            # Sample location from a flat prior, we'll add a pyro.factor statement later.
            internal_location = pyro.sample(
                "internal_location",
                dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1).mask(False),
            )

        # Note optimizers prefer numbers around 1, so we scale after the pyro.sample
        # statement, rather than in the distribution.
        internal_time = internal_time  # * self.Ne
        time = torch.zeros((self.num_nodes,))
        time[self.is_internal] = internal_time
        # Should we be Bayesian about migration scale, or should it be fixed?
        migration_scale = pyro.sample("migration_scale", dist.LogNormal(-5, 1))

        # Next add a factor for time gaps between parents and children.
        gap = time[self.parent] - time[self.child]
        with pyro.plate("edges", len(gap)):

            rate = (gap * self.span * self.mutation_rate).clamp(min=1e-8)
            pyro.sample(
                "mutations",
                dist.Poisson(rate),
                obs=self.mutations,
            )

            if self.migration_likelihood is None:
                location = torch.ones(self.ts.num_nodes)
            else:
                location = torch.cat([self.leaf_location, internal_location], 0)
                self.migration_likelihood(
                    self.parent, self.child, migration_scale, time, location
                )
        return time, gap, location, migration_scale


def euclidean_migration(parent, child, migration_scale, time, location):
    """
    Example::

        model = Model(
            ...,
            migration_likelihood=euclidean_migration
        )
    """
    gap = time[parent] - time[child]
    gap = gap.clamp(min=1e-10)
    # The following encodes that children migrate away from their parents
    # following brownian motion with rate migration_scale.
    parent_location = location.index_select(0, parent)
    child_location = location.index_select(0, child)
    # Note we need to .unsqueeze(-1) i.e. [..., None] the migration_scale
    # in case you want to draw multiple samples.
    migration_radius = migration_scale[..., None] * gap ** 0.5

    # Normalise distance
    distance = (child_location - parent_location).square().sum(-1).sqrt()
    pyro.sample(
        "migration",
        # Trying a heavy-tailed distribution
        dist.Exponential(1 / migration_radius),
        obs=distance,
    )


class WayPointMigration:
    """
    Example::

        model = Model(
            ...,
            migration_likelihood=WayPointMigration(
                transitions, waypoints, waypoint_radius
            )
        )
    """

    def __init__(self, transitions, waypoints, waypoint_radius):
        self.waypoint_radius = waypoint_radius
        self.waypoints = waypoints
        # This cheaply precomputes some matrices.
        self.matrix_exp = ApproximateMatrixExponential(self.transition)

    def __call__(self, parent, child, migration_scale, time, location):
        gap = time[parent] - time[child]
        gap = gap.clamp(min=1e-10)
        parent_location = location.index_select(0, parent)
        child_location = location.index_select(0, child)
        pyro.sample(
            "migration",
            WaypointDiffusion2D(
                source=parent_location,
                time=gap / migration_scale,  # FIXME is this right?
                radius=self.waypoint_radius,
                waypoints=self.waypoints,
                matrix_exp=self.matrix_exp,
            ),
            obs=child_location,
        )


def guide(ts, leaf_location, priors, Ne=10000, mutation_rate=1e-8, steps=1001):

    pyro.set_rng_seed(20210518)
    pyro.clear_param_store()

    model = NaiveModel(
        ts,
        leaf_location=leaf_location,
        prior=priors,
        Ne=Ne,
        mutation_rate=mutation_rate,
    )

    def init_loc_fn(site):
        if site["name"] == "internal_time":
            prior_init = np.einsum(
                "t,nt->n", priors.timepoints, (priors.grid_data[:])
            ) / np.sum(priors.grid_data[:], axis=1)
            internal_time = torch.as_tensor(prior_init, dtype=torch.float)  # / Ne
            return internal_time.clamp(min=0.1)
        # GEOGRAPHY.
        if site["name"] == "internal_location":
            initial_guess_loc = model.get_ancestral_geography(ts, leaf_location)
            return initial_guess_loc
        return init_to_median(site)

    guide = AutoNormal(
        model, init_scale=0.01, init_loc_fn=init_loc_fn
    )  # Mean field (fully Bayesian)
    optim = pyro.optim.ClippedAdam({"lr": 0.005, "lrd": 0.1 ** (1 / max(1, steps))})
    svi = SVI(model, guide, optim, Trace_ELBO())
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
                f"step {step} loss = {loss:0.5g}, "
                f"Migration scale= {migration_scale:0.3g}"
            )
    median = guide.median()
    pyro_time, gaps, location, migration_scale = poutine.condition(model, median)()
    return pyro_time, location, migration_scale, guide
