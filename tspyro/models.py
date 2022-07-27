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
from pyro.nn import PyroModule
from tspyro.diffusion import ApproximateMatrixExponential
from tspyro.diffusion import WaypointDiffusion2D
from tspyro.ops import CummaxUpTree


class BaseModel(PyroModule):
    def __init__(
        self, ts, *, Ne, prior, leaf_location=None, mutation_rate=1e-8, penalty=100.0, device=torch.device("cpu")
    ):
        super().__init__()
        nodes = ts.tables.nodes
        edges = ts.tables.edges
        self.is_leaf = torch.tensor((nodes.flags & 1).astype(bool), dtype=torch.bool)
        self.is_internal = ~self.is_leaf
        self.num_nodes = len(self.is_leaf)
        self.num_internal = self.is_internal.sum().item()
        self.device = device

        self.ts = ts

        self.parent = torch.tensor(edges.parent, dtype=torch.long, device=device)
        self.child = torch.tensor(edges.child, dtype=torch.long, device=device)
        self.span = torch.tensor(
            edges.right - edges.left, dtype=torch.get_default_dtype(), device=device
        )
        self.mutations = torch.tensor(
            self.get_mut_edges(), dtype=torch.get_default_dtype(), device=device
        )  # this is an int, but we optimise with float for pytorch

        self.penalty = float(penalty)
        self.Ne = float(Ne)
        self.mutation_rate = mutation_rate

        # conditional coalescent prior
        timepoints = torch.tensor(prior.timepoints, dtype=torch.get_default_dtype(), device=device)
        timepoints = timepoints.log1p()
        prior_grid_data = prior.grid_data[prior.row_lookup[ts.num_samples :]]  # noqa:
        grid_data = torch.as_tensor(prior_grid_data, dtype=torch.get_default_dtype(), device=device)
        grid_data = grid_data / grid_data.sum(1, True)
        self.prior_loc = torch.einsum("t,nt->n", timepoints, grid_data)
        self.prior_loc = self.prior_loc.nan_to_num(1)
        deltas = (timepoints - self.prior_loc[:, None]) ** 2
        self.prior_scale = torch.einsum("nt,nt->n", deltas, grid_data).sqrt()
        self.prior_scale = self.prior_scale.nan_to_num(1)
        self.leaf_location = leaf_location.to(device=device)

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

    def average_edges(self, parent_edges, locations, method="average"):
        parent = parent_edges[0]
        edges = parent_edges[1]

        child_spanfracs = list()
        child_lats = list()
        child_longs = list()

        for edge in edges:
            child_spanfracs.append(self.edge_span(edge))
            child_lats.append(locations[edge.child][0])
            child_longs.append(locations[edge.child][1])
        if method == "average":
            val = np.average(np.array([child_lats, child_longs]).T, axis=0)
        elif method == "geographic_mean":
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

    def get_ancestral_geography(
        self,
        ts,
        sample_locations,
        method="geographic_mean",
        device=torch.device("cpu"),
        show_progress=False,
    ):
        """
        Use dynamic programming to find approximate posterior to sample from
        """
        locations = np.zeros((ts.num_nodes, 2))
        locations[ts.samples()] = sample_locations
        fixed_nodes = set(ts.samples())

        # Iterate through the nodes via groupby on parent node
        for parent_edges in self.edges_by_parent_asc(ts):
            if parent_edges[0] not in fixed_nodes:
                parent, val = self.average_edges(parent_edges, locations, method=method)
                locations[parent] = val
        return torch.tensor(
            locations[ts.num_samples :],
            dtype=torch.get_default_dtype(),
            device=device,  # noqa: E203
        )


class NaiveModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.migration_likelihood = kwargs.pop("migration_likelihood", None)
        self.location_model = kwargs.pop("location_model", mean_field_location)
        super().__init__(*args, **kwargs)

    def forward(self):
        # First sample times from an improper uniform distribution which we denote
        # via .mask(False). Note only the internal nodes are sampled; leaves are
        # fixed at zero.
        # Note this isn't a coalescent prior, but some are available at:
        # https://docs.pyro.ai/en/stable/_modules/pyro/distributions/coalescent.html
        with pyro.plate("internal_nodes", self.num_internal):
            internal_time = pyro.sample(
                "internal_time",
                dist.LogNormal(self.prior_loc, self.prior_scale).mask(
                    True
                ),  # False turns off prior but uses it for initialisation
            )  # internal time is modelled in logspace

            if self.leaf_location is not None:
                internal_location = self.location_model()

        # Note optimizers prefer numbers around 1, so we scale after the pyro.sample
        # statement, rather than in the distribution.
        internal_time = internal_time  # * self.Ne
        time = torch.zeros(
            internal_time.shape[:-1] + (self.num_nodes,), device=self.device
        )
        time[..., self.is_internal] = internal_time
        # Should we be Bayesian about migration scale, or should it be fixed?
        migration_scale = pyro.sample("migration_scale", dist.LogNormal(0, 4))

        # Next add a factor for time gaps between parents and children.
        gap = time[..., self.parent] - time[..., self.child]
        with pyro.plate("edges", gap.size(-1)):
            # Penalize gaps that are less than 1.
            clamped_gap = gap.clamp(min=1)
            # TODO should we multiply this by e.g. 0.1
            pyro.factor("gap_constraint", gap - clamped_gap)

            rate = (clamped_gap * self.span * self.mutation_rate).clamp(min=1e-8)
            pyro.sample(
                "mutations",
                dist.Poisson(rate),
                obs=self.mutations,
            )

            if self.migration_likelihood is None:
                location = torch.ones(self.ts.num_nodes, device=device)
            else:
                # We need to expand leaf_location because internal_location may
                # be vectorized over monte carlo samples.
                batch_shape = internal_location.shape[:-2]
                location = torch.cat(
                    [
                        self.leaf_location.expand(batch_shape + (-1, -1)),
                        internal_location,
                    ],
                    -2,
                )
                self.migration_likelihood(
                    self.parent, self.child, migration_scale, time, location
                )
        return time, gap, location, migration_scale


class TimeDiffModel(BaseModel):
    """
    This is like NaiveModel but with a time parameterization that has better
    geometry, in particular, we don't need to include hard constraints for
    positive time gaps.
    """

    def __init__(self, *args, **kwargs):
        self.migration_likelihood = kwargs.pop("migration_likelihood", None)
        self.location_model = kwargs.pop("location_model", mean_field_location)
        ts = args[0] if args else kwargs["ts"]
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            # Initialize the prior time differences.
            self.cumsum_up_tree = CummaxUpTree(ts)
            time = torch.zeros(self.num_nodes)
            time[..., self.is_internal] = self.prior_loc.exp()
            diff = self.cumsum_up_tree.inverse(time).clamp(min=0.1)
            self.prior_diff_loc = diff[self.is_internal].log()

    def forward(self):
        edges_plate = pyro.plate("edges", len(self.parent), dim=-1)
        internal_nodes_plate = pyro.plate("internal_nodes", self.num_internal, dim=-1)

        # Integrate times up the tree, saving result via pyro.deterministic.
        with internal_nodes_plate:
            internal_diff = pyro.sample(
                "internal_diff",
                dist.LogNormal(self.prior_diff_loc, self.prior_scale),
            )
            batch_shape = internal_diff.shape[:-1]
        diff = torch.zeros(batch_shape + self.is_internal.shape)
        diff[..., self.is_internal] = internal_diff
        time = self.cumsum_up_tree(diff)
        pyro.deterministic("internal_time", time.detach()[..., self.is_internal])

        # Mutation part of the model.
        gap = (time[..., self.parent] - time[..., self.child]).clamp(min=1)
        with edges_plate:
            rate = (gap * self.span * self.mutation_rate).clamp(min=1e-8)
            pyro.sample("mutations", dist.Poisson(rate), obs=self.mutations)

        # Geographic part of the model.
        location = None
        migration_scale = None
        if self.migration_likelihood is not None:
            migration_scale = pyro.sample("migration_scale", dist.LogNormal(0, 4))
            with internal_nodes_plate:
                internal_location = self.location_model()
            location = torch.cat(
                [
                    self.leaf_location.expand(batch_shape + (-1, -1)),
                    internal_location,
                ],
                -2,
            )
            with edges_plate:
                self.migration_likelihood(
                    self.parent, self.child, migration_scale, time, location
                )

        return time, gap, location, migration_scale


def mean_field_location():
    """
    To create a conditioned version of this model, use::

        location = pyro.condition(
            data={"internal_location": true_locations}
        )(mean_field_location)
    """
    # Sample location from a flat prior, we'll add a pyro.factor statement later.
    return pyro.sample(
        "internal_location",
        dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1).mask(False),
    )


# TODO implement class ReparamLocation from ReparamModel.
class ReparamLocation:
    """
    Example::

        model = Model(
            ...,
            location_model=ReparamLocation()
    """

    def __init__(self, ts, leaf_location):
        dtype = torch.get_default_dtype()
        D = 2  # assume 2D for now
        N = ts.num_nodes
        L = len(leaf_location)

        # These are coefficients in an affine reparametrization:
        #   location = baseline_0 + baseline_1 @ delta  # i.e. constant + matmul
        # where the delta matrix of shape (N,D) is our new latent variable.
        baseline_0 = torch.zeros(N, D)  # intercept
        baseline_0[:L] = leaf_location
        baseline_1 = torch.zeros(N, N)  # slope
        baseline_1[L:, L:] = torch.eye(N - L)  # each location depends on its delta
        times = torch.tensor(
            ts.tables.nodes.time, dtype=dtype
        )  # assume times are fixed

        for parent, edges in itertools.groupby(
            ts.edges(), operator.attrgetter("parent")
        ):
            children = [e.child for e in edges]
            assert children
            children = torch.tensor(children)

            # Compute a 1/gap weighted average of child locations.
            gaps = times[parent] - times[children]
            assert (gaps > 0).all()
            weights = 1 / gaps  # Brownian weights
            weights /= weights.sum()  # normalize

            # Parent's location is weighted sum of child locations, plus delta.
            baseline_0[parent] = (weights[:, None] * baseline_0[children]).sum(0)
            baseline_1[parent] += (weights[:, None] * baseline_1[children]).sum(0)
        # Restrict to internal nodes.
        baseline_0 = baseline_0[L:].clone()
        baseline_1 = baseline_1[L:, L:].clone()
        self.baseline = baseline_0, baseline_1

    def __call__(self):
        # GEOGRAPHY.
        # Locations will be parametrized as difference from a baseline, where the
        # baseline location of a parent is a weighted sum of child locations.
        # Sample locations from a flat prior, we'll add a pyro.factor statement later.
        internal_delta = pyro.sample(
            "internal_delta",
            dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1).mask(False),
        )  # [self.num_internal, 2]
        baseline_0, baseline_1 = self.baseline
        internal_location = baseline_0 + baseline_1 @ internal_delta
        return internal_location


def great_circle(lon1, lat1, lon2, lat2):
    lon1 = torch.deg2rad(lon1)
    lat1 = torch.deg2rad(lat1)
    lon2 = torch.deg2rad(lon2)
    lat2 = torch.deg2rad(lat2)
    return 6371 * (
        torch.acos(
            torch.sin(lat1) * torch.sin(lat2)
            + torch.cos(lat1) * torch.cos(lat2) * torch.cos(lon1 - lon2)
        )
    )


def great_secant(lon1, lat1, lon2, lat2):
    lon = torch.deg2rad(torch.stack([lon1, lon2]))
    lat = torch.deg2rad(torch.stack([lat1, lat2]))
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    r2 = (x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2 + (z[0] - z[1]) ** 2
    return r2.clamp(min=1e-10).sqrt()


def euclidean_migration(parent, child, migration_scale, time, location):
    """
    Example::

        model = Model(
            ...,
            migration_likelihood=euclidean_migration
        )
    """
    gap = time[..., parent] - time[..., child]  # in units of generations
    gap = gap.clamp(
        min=1
    )  # avoid incorrect ordering of parents/children due to unconstrained
    # time latent variable
    # The following encodes that children migrate away from their parents
    # following approximately Brownian motion with rate migration_scale.
    parent_location = location.index_select(-2, parent)
    child_location = location.index_select(-2, child)
    # Note we need to .unsqueeze(-1) i.e. [..., None] the migration_scale
    # in case you want to draw multiple samples.
    migration_radius = migration_scale[..., None] * gap**0.5

    # Assume migration folows a bivariate normal distribution, so that
    # distance follows a Gamma(2,-) distribution.  While a more theoretically
    # sound model might replace the Brownian motion's Wiener process with a
    # heavier tailed Levy stable process, the Stable distribution's tail is so
    # heavy that inference becomes intractable.  To give our unimodal
    # variational posterior a chance of finding the right mode, we use a
    # log-concave likelihood with tails heavier than Normal but lighter than
    # Stable.  An alternative might be to anneal tail weight.

    if False:
        distance = torch.linalg.norm(child_location - parent_location, dim=-1, ord=2)
        distance = distance.clamp(min=1e-6)
        pyro.sample(
            "migration",
            dist.Gamma(2, 1 / migration_radius),
            obs=distance,
        )
    # This is equivalent to
    else:
        pyro.sample(
            "migration",
            dist.Normal(parent_location, migration_radius[..., None]).to_event(1),
            obs=child_location,
        )


def latlong_to_xyz(latlong: torch.Tensor) -> torch.Tensor:
    lat, long = latlong.unbind(-1)
    x = lat.cos() * long.cos()
    y = lat.cos() * long.sin()
    z = lat.sin()
    return torch.cat([x, y, z], dim=-1)


def spherical_migration(parent, child, migration_scale, time, location):
    """
    Example::

        model = Model(
            ...,
            migration_likelihood=spherical_migration
        )
    """
    gap = time[parent] - time[child]  # in units of generations
    gap = gap.clamp(
        min=1
    )  # avoid incorrect ordering of parents/children due to unconstrained
    # time latent variable
    # The following encodes that children migrate away from their parents
    # following approximately Brownian motion with rate migration_scale.
    parent_location = location.index_select(-2, parent)
    child_location = location.index_select(-2, child)
    # Note we need to .unsqueeze(-1) i.e. [..., None] the migration_scale
    # in case you want to draw multiple samples.
    migration_radius = migration_scale[..., None] * gap**0.5

    # Assume migration folows a bivariate Laplace distribution, so that
    # distance follows a Gamma(2,-) distribution.  While a more theoretically
    # sound model might replace the Brownian motion's Wiener process with a
    # heavier tailed Levy stable process, the Stable distribution's tail is so
    # heavy that inference becomes intractable.  To give our unimodal
    # variational posterior a chance of finding the right mode, we use a
    # log-concave likelihood with tails heavier than Normal but lighter than
    # Stable.  An alternative might be to anneal tail weight.
    child_xyz = latlong_to_xyz(child_location)
    parent_xyz = latlong_to_xyz(parent_location)
    distance = torch.linalg.norm(child_xyz - parent_xyz, dim=-1, ord=2)
    pyro.sample(
        "migration",
        dist.Gamma(2, 1 / migration_radius),
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

    def __init__(self, transitions, waypoints, waypoint_radius, device=torch.device("cpu")):
        self.waypoint_radius = waypoint_radius
        self.waypoints = waypoints
        self.device = device
        # This cheaply precomputes some matrices.
        self.matrix_exp = ApproximateMatrixExponential(
            transitions, max_time_step=1e6
        )  # TODO: need to fix max time step

    def __call__(self, parent, child, migration_scale, time, location):
        gap = (time[parent] - time[child]).clamp(min=1)
        parent_location = location.index_select(0, parent)
        child_location = location.index_select(0, child)
        pyro.sample(
            "migration",
            WaypointDiffusion2D(
                source=parent_location,
                time=gap * migration_scale,
                radius=self.waypoint_radius,
                waypoints=self.waypoints,
                matrix_exp=self.matrix_exp,
                device=self.device
            ),
            obs=child_location,
        )


def fit_guide(
    ts,
    leaf_location,
    priors,
    init_loc=None,
    Ne=10000,
    mutation_rate=1e-8,
    migration_scale_init=1,
    steps=1001,
    Model=NaiveModel,
    migration_likelihood=None,
    location_model=mean_field_location,
    learning_rate=0.005,
    learning_rate_decay=0.1,
    log_every=100,
    device=None,
):
    assert isinstance(Model, type)

    pyro.set_rng_seed(20210518)
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
        device=device
    )
    model = model.to(device=device)

    def init_loc_fn(site):
        # TIME
        if site["name"] == "internal_time":
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
                initial_guess_loc = model.get_ancestral_geography(
                    ts, leaf_location, device=device
                )
            return initial_guess_loc
        if site["name"] == "internal_delta":
            return torch.zeros(site["fn"].shape())
        if site["name"] == "migration_scale":
            return torch.tensor(float(migration_scale_init), device=device)
        raise NotImplementedError("Missing init for {}".format(site["name"]))

    guide = AutoNormal(
        model, init_scale=0.01, init_loc_fn=init_loc_fn
    )  # Mean field (fully Bayesian)
    optim = pyro.optim.ClippedAdam(
        {"lr": learning_rate, "lrd": learning_rate_decay ** (1 / max(1, steps))}
    )
    svi = SVI(model, guide, optim, Trace_ELBO())
    guide()  # initialises the guide
    losses = []
    migration_scales = []
    for step in range(steps):
        loss = svi.step() / ts.num_nodes
        losses.append(loss)
        if step % log_every == 0:
            with torch.no_grad():
                median = (
                    guide.median()
                )  # assess convergence of migration scale parameter
                try:
                    migration_scale = float(median["migration_scale"])
                    migration_scales.append(migration_scale)
                except KeyError:
                    migration_scale = None
                    print("Migration scale is fixed")
            print(
                f"step {step} loss = {loss:0.5g}, "
                f"Migration scale= {migration_scale}"
            )
    median = guide.median()
    pyro_time, gaps, location, migration_scale = poutine.condition(model, median)()
    return pyro_time, location, migration_scale, guide, losses


def single_chromosome_reaction_model(
    reproduction_op: torch.Tensor,
    leaf_times: torch.Tensor,
    leaf_clusters: torch.Tensor,
    num_time_steps: int,
):
    assert reproduction_op.dtype == torch.float
    assert leaf_times.dtype == torch.long
    assert leaf_clusters.dtype == torch.long
    T = num_time_steps
    C = len(reproduction_op)
    N = len(leaf_times)
    assert reproduction_op.shape == (C, C, C)
    assert leaf_times.shape == (N,)
    assert leaf_clusters.shape == (N,)
    assert leaf_times.max().item() < num_time_steps
    time_plate = pyro.plate("time", T, dim=-2)
    step_plate = pyro.plate("step", T - 1, dim=-2)
    cluster_plate = pyro.plate("cluster", C, dim=-1)
    leaf_plate = pyro.plate("leaves", N, dim=-1)

    # Sample density from an improper prior.
    # Consider instead using an informative prior.
    # Consider using HaarReparam(dim=-3) or DiscreteCosineReparam(dim=-3).
    with time_plate, cluster_plate, pyro.mask(mask=False):
        density = pyro.sample("density", dist.Exponential(1))

    # Reaction factor.
    # Note technically there are dependencies across cluster_plate.
    with step_plate, cluster_plate:
        parent_dist = density[:-1]
        child_dist = density[1:]
        prediction = torch.einsum(
            "tc,td,cde->te",
            parent_dist,  # [T,C]
            parent_dist,  # [T,C]
            reproduction_op,  # [C,C,C] Applies crossover.
        )  # [T,C]
        # Renormalize after selection_op.
        prediction = prediction / prediction.sum(-1, True)
        pyro.sample(
            "reaction",
            dist.Normal(prediction, prediction.sqrt()),  # approximate multinomial
            obs=child_dist,
        )

    # Observation of samples with known (time, genome).
    with leaf_plate:
        pyro.factor("obs", density[leaf_times, leaf_clusters].log())
