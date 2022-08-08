import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from tspyro.diffusion import ApproximateMatrixExponential
from tspyro.diffusion import WaypointDiffusion2D
from tspyro.ops import (
    CummaxUpTree,
    edges_by_parent_asc,
    get_mut_edges,
    latlong_to_xyz,
)
import tsdate


class BaseModel(PyroModule):
    def __init__(
        self,
        ts,
        *,
        Ne,
        leaf_location=None,
        mutation_rate=1e-8,
        penalty=100.0,
        progress=False,
        gap_prefactor=1.0,
        gap_exponent=1.0,
    ):
        super().__init__()
        nodes = ts.tables.nodes
        edges = ts.tables.edges
        self.is_leaf = torch.tensor((nodes.flags & 1).astype(bool), dtype=torch.bool)
        self.is_internal = ~self.is_leaf
        self.num_nodes = len(self.is_leaf)
        self.num_internal = self.is_internal.sum().item()

        self.ts = ts

        self.gap_prefactor = gap_prefactor
        self.gap_exponent = gap_exponent

        self.parent = torch.tensor(edges.parent, dtype=torch.long)
        self.child = torch.tensor(edges.child, dtype=torch.long)
        self.span = torch.tensor(
            edges.right - edges.left, dtype=torch.get_default_dtype()
        )
        self.mutations = torch.tensor(
            get_mut_edges(self.ts), dtype=torch.get_default_dtype()
        )  # this is an int, but we optimise with float for pytorch

        self.penalty = float(penalty)
        self.Ne = float(Ne)
        self.mutation_rate = mutation_rate

        # conditional coalescent prior
        nodes_to_date = (~self.is_leaf).data.cpu().numpy()
        span_data = tsdate.prior.SpansBySamples(ts, progress=progress)
        approximate_prior_size = 1000
        prior_distribution = "lognorm"
        base_priors = tsdate.prior.ConditionalCoalescentTimes(
            approximate_prior_size, self.Ne, prior_distribution, progress=progress
        )
        base_priors.add(ts.num_samples, True)

        for total_fixed in span_data.total_fixed_at_0_counts:
            # For missing data: trees vary in total fixed node count => have
            # different priors
            if total_fixed > 0:
                base_priors.add(total_fixed, True)

        prior_params = base_priors.get_mixture_prior_params(span_data)[:-1]
        self.prior_scale = torch.sqrt(
            torch.tensor(
                prior_params[nodes_to_date, 1], dtype=torch.get_default_dtype()
            )
        )
        self.prior_loc = torch.tensor(
                prior_params[nodes_to_date, 0], dtype=torch.get_default_dtype()
            )

        self.leaf_location = leaf_location


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
        time = torch.zeros(internal_time.shape[:-1] + (self.num_nodes,)).type_as(internal_time)
        time[..., self.is_internal] = internal_time

        migration_scale = None
        if self.migration_likelihood is not None:
            if self.migration_likelihood.__name__ == "euclidean_migration":
                migration_scale = pyro.sample("migration_scale", dist.LogNormal(0, 4))

        # Next add a factor for time gaps between parents and children.
        gap = time[..., self.parent] - time[..., self.child]
        with pyro.plate("edges", gap.size(-1)):
            # Penalize gaps that are less than 1.
            clamped_gap = gap.clamp(min=1)
            # TODO should we multiply this by e.g. 0.1
            prefactor, exponent = self.gap_prefactor, self.gap_exponent
            if exponent != 1.0:
                pyro.factor("gap_constraint", prefactor * (gap - clamped_gap).pow(exponent))
            else:
                pyro.factor("gap_constraint", prefactor * (gap - clamped_gap))

            rate = (clamped_gap * self.span * self.mutation_rate).clamp(min=1e-8)
            pyro.sample(
                "mutations",
                dist.Poisson(rate),
                obs=self.mutations,
            )

            if self.migration_likelihood is None:
                location = torch.ones(self.ts.num_nodes)
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
                migration_scale = self.migration_likelihood(
                    self.parent, self.child, migration_scale, time, location
                )
                if self.migration_likelihood.__name__ == 'marginal_euclidean_migration':
                    pyro.get_param_store()['migration_scale'] = migration_scale.data

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

        for parent, edges in edges_by_parent_asc(ts):
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


def marginal_euclidean_migration(parent, child, migration_scale, time, location):
    """
    """
    gap = time[..., parent] - time[..., child]  # in units of generations
    gap = gap.clamp(
        min=1
    )  # avoid incorrect ordering of parents/children due to unconstrained
    parent_location = location.index_select(-2, parent)
    child_location = location.index_select(-2, child)
    delta_loc_sq = (parent_location - child_location).pow(2.0).mean(-1)
    migration_scale = (delta_loc_sq / gap).mean().sqrt()

    pyro.sample(
        "migration",
        dist.Normal(parent_location, migration_scale[..., None]).to_event(1),
        obs=child_location,
    )

    return migration_scale


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
    migration_radius = migration_scale[..., None] * gap ** 0.5

    pyro.sample(
        "migration",
        dist.Normal(parent_location, migration_radius[..., None]).to_event(1),
        obs=child_location,
    )

    return migration_scale


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
    migration_radius = migration_scale[..., None] * gap ** 0.5

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

    def __init__(self, transitions, waypoints, waypoint_radius):
        self.waypoint_radius = waypoint_radius
        self.waypoints = waypoints
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
            ),
            obs=child_location,
        )
