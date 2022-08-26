import time

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
import numpy as np
from tspyro.diffusion import ApproximateMatrixExponential
from tspyro.diffusion import WaypointDiffusion2D
from tspyro.ops import (
    CummaxUpTree,
    CumlogsumexpUpTree,
    edges_by_parent_asc,
    get_mut_edges,
    latlong_to_xyz,
)
import tsdate
from util import get_metadata
from tspyro.ops import get_ancestral_geography


class BaseModel(PyroModule):
    def __init__(
        self,
        ts,
        *,
        Ne,
        leaf_location=None,
        mutation_rate=1e-8,
        progress=False,
        gap_prefactor=1.0,
        gap_exponent=1.0,
        min_gap=1.0,
        compute_time_prior=True
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
        self.min_gap = min_gap

        self.parent = torch.tensor(edges.parent, dtype=torch.long)
        self.child = torch.tensor(edges.child, dtype=torch.long)
        self.span = torch.tensor(
            edges.right - edges.left, dtype=torch.get_default_dtype()
        )
        self.mutations = torch.tensor(
            get_mut_edges(self.ts), dtype=torch.get_default_dtype()
        )  # this is an int, but we optimise with float for pytorch

        self.Ne = float(Ne)
        self.mutation_rate = mutation_rate
        self.leaf_location = leaf_location

        if not compute_time_prior:
            return

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


class NaiveModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.migration_likelihood = kwargs.pop("migration_likelihood", None)
        self.location_model = kwargs.pop("location_model", mean_field_location)
        self.poisson_likelihood = kwargs.pop("poisson_likelihood", True)
        super().__init__(*args, **kwargs)
        self.leaf_times = torch.as_tensor(self.ts.tables.nodes.time[self.is_leaf.data.cpu().numpy()],
                                          dtype=torch.get_default_dtype())

    def forward(self):
        # First sample times from an improper uniform distribution which we denote
        # via .mask(False). Note only the internal nodes are sampled; leaves are
        # fixed at zero.
        # Note this isn't a coalescent prior, but some are available at:
        # https://docs.pyro.ai/en/stable/_modules/pyro/distributions/coalescent.html
        #log_Ne = pyro.param("log_Ne", torch.tensor(0.0))
        #if torch.rand(1).item() < 0.003:
        #    print("log_Ne",log_Ne.item())

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
        # Fix time of samples
        time[..., self.is_leaf] = self.leaf_times
        time[..., self.is_internal] = internal_time

        migration_scale = None
        if self.migration_likelihood is not None:
            if self.migration_likelihood.__name__ == "euclidean_migration":
                migration_scale = pyro.sample("migration_scale", dist.LogNormal(0, 4))

        # Next add a factor for time gaps between parents and children.
        gap = time[..., self.parent] - time[..., self.child]
        with pyro.plate("edges", gap.size(-1)):
            # Penalize gaps that are less than 1.
            clamped_gap = gap.clamp(min=self.min_gap)
            # TODO should we multiply this by e.g. 0.1
            prefactor, exponent = self.gap_prefactor, self.gap_exponent
            if exponent != 1.0 and exponent != 0.0:
                pyro.factor("gap_constraint", -prefactor * (gap - clamped_gap).abs().clamp(min=1.0e-8).pow(exponent))
            elif exponent == 0.0:
                clamped_gap = gap.clamp(min=-1.0e-8)
                pyro.factor("gap_constraint", -prefactor * (-gap + clamped_gap + 1.0).log())
            else:
                pyro.factor("gap_constraint", prefactor * (gap - clamped_gap))

            rate = (clamped_gap * self.span * self.mutation_rate).clamp(min=1e-8)
            if self.poisson_likelihood:
                pyro.sample(
                    "mutations",
                    dist.Poisson(rate),
                    obs=self.mutations,
                )
            else:
                total_count = pyro.param("total_count", torch.tensor(20.0),
                    constraint=torch.distributions.constraints.positive)
                rate_sigma = pyro.param("rate_sigma", torch.tensor(1.0e-2),
                    constraint=torch.distributions.constraints.positive)
                logits = rate.log() - total_count.log() - 0.5 * rate_sigma.pow(2.0)
                pyro.sample(
                    "mutations",
                    dist.LogNormalNegativeBinomial(total_count=total_count, logits=logits,
                        multiplicative_noise_scale=rate_sigma),
                    obs=self.mutations)

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
        self.scale_factor = kwargs.pop("scale_factor", 1.0)
        ts = args[0] if args else kwargs["ts"]
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            # Initialize the prior time differences.
            self.cumsum_up_tree = CumlogsumexpUpTree(ts, scale_factor=self.scale_factor)
            #self.cumsum_up_tree = CummaxUpTree(ts)
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


class ConditionedTimesSimplifiedModel(torch.nn.Module):
    def __init__(self,
                 ts,
                 time_cutoff=50.0,
                 strategy='fill',
                 dtype=torch.float64,
                 device=torch.device('cpu')):

        super().__init__()
        nodes = ts.tables.nodes
        edges = ts.tables.edges
        self.ts = ts
        self.device = device
        self.dtype = dtype

        self.strategy = strategy
        assert strategy in ['fill', 'sever']

        self.times = torch.as_tensor(ts.tables.nodes.time, dtype=self.dtype, device=device)
        self.parent = torch.tensor(edges.parent, dtype=torch.long, device=device)
        self.child = torch.tensor(edges.child, dtype=torch.long, device=device)

        self.observed = torch.tensor((nodes.flags & 1).astype(bool), dtype=torch.bool, device=device)
        self.unobserved = ~self.observed
        self.num_unobserved = int(self.unobserved.sum().item())
        self.num_observed = int(self.observed.sum().item())
        self.num_nodes = len(self.observed)
        self.num_edges = self.parent.size(0)
        assert self.num_nodes == self.num_unobserved + self.num_observed

        print("num edges: {}   num nodes: {}".format(self.num_edges, self.num_nodes))
        print("num unobserved nodes: {}   num observed nodes: {}".format(self.num_unobserved, self.num_observed))
        print("time min/max: {:.1f} / {:.1f}".format(self.times.min().item(), self.times.max().item()))

        self.locations = torch.as_tensor(get_metadata(self.ts)[0], dtype=self.dtype, device=device)
        self.nan_locations = self.locations.isnan().sum(-1) > 0
        self.min_time_cutoff = self.times[self.nan_locations].min().item()
        self.time_cutoff = min(time_cutoff, self.min_time_cutoff)
        print("Using a time cutoff of {:.1f} with {} strategy".format(self.time_cutoff, self.strategy))

        # compute heuristic location estimates
        self.initial_loc = torch.zeros(self.num_nodes, 2, dtype=dtype, device=device)
        initial_loc = get_ancestral_geography(self.ts, self.locations[self.observed].data.cpu().numpy()).type_as(self.locations)
        self.initial_loc[self.unobserved] = initial_loc
        assert self.initial_loc.isnan().sum().item() == 0.0

        # define dividing temporal boundary characterized by time_cutoff
        self.old_unobserved = (self.times >= time_cutoff) & self.unobserved
        num_old = int(self.old_unobserved.sum().item())
        print("Filling in {} unobserved nodes with heuristic locations".format(num_old))

        self.observed = self.observed | self.old_unobserved
        self.unobserved = ~self.observed
        self.num_unobserved = int(self.unobserved.sum().item())
        self.num_observed = int(self.observed.sum().item())
        assert self.num_nodes == self.num_unobserved + self.num_observed
        print("num_unobserved", self.num_unobserved, "num_observed", self.num_observed)

        # fill in old unobserved locations with heuristic guess (w/ old defined by time_cutoff)
        self.locations[self.old_unobserved] = self.initial_loc[self.old_unobserved]
        # optionally sever edges that contain old unobserved locations (w/ old defined by time_cutoff)
        if self.strategy == 'sever':
            old_nodes = set(torch.arange(self.num_nodes)[self.old_unobserved].tolist())
            old_parent = torch.tensor([int(par.item()) in old_nodes for par in self.parent], dtype=torch.bool)
            old_child = torch.tensor([int(chi.item()) in old_nodes for chi in self.child], dtype=torch.bool)
            edges_to_keep = ~(old_parent | old_child)
            self.parent = self.parent[edges_to_keep]
            self.child = self.child[edges_to_keep]
            print("Number of edges after severing: {}".format(self.parent.size(0)))

        self.parent_observed = self.observed[self.parent]
        self.child_observed = self.observed[self.child]
        self.parent_unobserved = self.unobserved[self.parent]
        self.child_unobserved = self.unobserved[self.child]

        self.doubly_unobserved = self.parent_unobserved & self.child_unobserved
        self.doubly_observed = self.parent_observed & self.child_observed
        self.singly_observed_child = self.parent_unobserved & self.child_observed
        self.singly_observed_parent = self.parent_observed & self.child_unobserved
        print("doubly_unobserved: ", self.doubly_unobserved.sum().item(),
              "  singly_observed: ", self.singly_observed_child.sum().item() + self.singly_observed_parent.sum().item(),
              "  doubly_observed: ", self.doubly_observed.sum().item())
        assert (self.doubly_unobserved + self.singly_observed_child + self.singly_observed_parent +
                self.doubly_observed).sum().item() == self.parent.size(0)

        self.double_edges_parent = self.parent[self.doubly_unobserved]
        self.double_edges_child = self.child[self.doubly_unobserved]
        self.single_edges_child_parent = self.parent[self.singly_observed_child]
        self.single_edges_child_child = self.child[self.singly_observed_child]
        self.single_edges_parent_parent = self.parent[self.singly_observed_parent]
        self.single_edges_parent_child = self.child[self.singly_observed_parent]

        self.edge_times = (self.times[self.parent] - self.times[self.child]).clamp(min=1.0)
        self.sqrt_edge_times = self.edge_times ** 0.5

        self.sqrt_edge_times_doubly_unobserved = self.sqrt_edge_times[self.doubly_unobserved]
        self.sqrt_edge_times_singly_observed_child = self.sqrt_edge_times[self.singly_observed_child]
        self.sqrt_edge_times_singly_observed_parent = self.sqrt_edge_times[self.singly_observed_parent]

    def forward(self):
        migration_scale = torch.tensor(0.30, dtype=self.dtype, device=self.device)

        with pyro.plate("unobserved_locs", self.num_unobserved):
            internal_locs = pyro.sample("internal_location",
                                        dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1).mask(False))
            assert internal_locs.ndim == 2
            locs = self.locations.clone()
            locs[self.unobserved] = internal_locs

        with pyro.plate("double_edges", self.double_edges_parent.size(0)):
            pyro.sample("migration1",
                dist.Normal(locs.index_select(-2, self.double_edges_parent),
                            migration_scale * self.sqrt_edge_times_doubly_unobserved.unsqueeze(-1)).to_event(1),
                obs=locs.index_select(-2, self.double_edges_child))

        with pyro.plate("single_edges_child", self.single_edges_child_child.size(0)):
            pyro.sample("migration2",
                dist.Normal(locs.index_select(-2, self.single_edges_child_parent),
                            migration_scale * self.sqrt_edge_times_singly_observed_child.unsqueeze(-1)).to_event(1),
                obs=locs.index_select(-2, self.single_edges_child_child))

        with pyro.plate("single_edges_parent", self.single_edges_parent_parent.size(0)):
            pyro.sample("migration3",
                dist.Normal(locs.index_select(-2, self.single_edges_parent_parent),
                            migration_scale * self.sqrt_edge_times_singly_observed_parent.unsqueeze(-1)).to_event(1),
                obs=locs.index_select(-2, self.single_edges_parent_child))

        return self.times, None, locs, migration_scale



class ConditionedTimesNaiveModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.migration_likelihood = kwargs.pop("migration_likelihood", None)
        self.location_model = kwargs.pop("location_model", mean_field_location)
        time_mask = kwargs.pop("time_mask", None)
        self.time_cutoff = kwargs.pop("time_cutoff", 100.0)
        heuristic_loc = kwargs.pop("heuristic_loc", None)
        super().__init__(*args, compute_time_prior=False, **kwargs)
        self.time_mask = time_mask
        self.heuristic_loc = heuristic_loc.clone()
        self.internal_times = torch.as_tensor(self.ts.tables.nodes.time[self.is_internal.data.cpu().numpy()],
                                              dtype=torch.get_default_dtype())
        self.internal_time_mask = self.internal_times >= self.time_cutoff
        self.leaf_times = torch.as_tensor(self.ts.tables.nodes.time[self.is_leaf.data.cpu().numpy()],
                                          dtype=torch.get_default_dtype())
        self.times = torch.as_tensor(self.ts.tables.nodes.time,
                                     dtype=torch.get_default_dtype())

        true_locations = torch.as_tensor(get_metadata(self.ts)[0], dtype=torch.get_default_dtype())
        bins = [0, 10.0, 30.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
        print("[Empirical migration scales]")
        for left, right in zip(bins[:-1], bins[1:]):
            gap = self.times[..., self.parent] - self.times[..., self.child]  # in units of generations
            gap = gap.clamp(min=1.0)
            parent_location = true_locations.index_select(-2, self.parent)  # num_particles num_edges 2
            child_location = true_locations.index_select(-2, self.child)
            delta_loc_sq = (parent_location - child_location).pow(2.0).mean(-1)  # num_particles num_edges
            time_mask = (self.times[self.parent] >= left) & (self.times[self.parent] < right) \
                & (~delta_loc_sq.isnan())
            delta_loc_sq[delta_loc_sq.isnan()] = 0.0

            scale = ((delta_loc_sq / gap) * time_mask.type_as(gap)).sum(-1).mean(0) / time_mask.sum().item()
            print("[{}, {}] \t scale: {:.4f}   num_edges: {}".format(int(left), int(right), scale,
                                                                  int(time_mask.sum().item())))

    def forward(self):
        with pyro.plate("internal_nodes", self.num_internal):
            internal_location = self.location_model()
            internal_location[..., self.internal_time_mask, :] = self.heuristic_loc[self.internal_time_mask, :]

        migration_scale = None
        assert self.migration_likelihood is not None
        if self.migration_likelihood.__name__ == "euclidean_migration":
            migration_scale = pyro.sample("migration_scale", dist.LogNormal(0, 4))

        with pyro.plate("edges", self.parent.size(-1)):
            batch_shape = internal_location.shape[:-2]
            location = torch.cat(
                [
                    self.leaf_location.expand(batch_shape + (-1, -1)),
                    internal_location,
                ],
                -2,
            )
            migration_scale = self.migration_likelihood(
                self.parent, self.child, migration_scale, self.times, location, self.time_mask
            )
            if self.migration_likelihood.__name__ == 'marginal_euclidean_migration':
                pyro.get_param_store()['migration_scale'] = migration_scale.data

        return self.times, None, location, migration_scale


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


def marginal_euclidean_migration(parent, child, migration_scale, time, location, time_mask):
    """
    """
    gap = time[..., parent] - time[..., child]  # in units of generations
    gap = gap.clamp(min=1)  # num_edges
    parent_location = location.index_select(-2, parent)  # num_particles num_edges 2
    child_location = location.index_select(-2, child)
    if 0:
        num_observed_pairs = time_mask.sum().item()
        delta_loc_sq = (parent_location - child_location).pow(2.0).mean(-1)  # num_particles num_edges
        migration_scale = ((delta_loc_sq / gap) * time_mask.type_as(gap)).sum(-1).mean(0) / num_observed_pairs
        migration_scale_median = (delta_loc_sq / gap)[time_mask].median().item()
        migration_scale_mean = (delta_loc_sq / gap)[time_mask].sort()[0]
        migration_scale_mean75 = migration_scale_mean[:int(75 * migration_scale_mean.size(0) // 100)].mean().item()
        migration_scale_mean90 = migration_scale_mean[:int(90 * migration_scale_mean.size(0) // 100)].mean().item()
        migration_scale_mean95 = migration_scale_mean[:int(95 * migration_scale_mean.size(0) // 100)].mean().item()
        migration_scale_mean99 = migration_scale_mean[:int(99 * migration_scale_mean.size(0) // 100)].mean().item()
        if torch.rand(1).item() < 0.0:
            print("migration_scale_median", migration_scale_median)
            print("quantiles", migration_scale_mean75, migration_scale_mean90, migration_scale_mean95, migration_scale_mean99)

    else:
        migration_scale = torch.tensor(0.3)
    migration_scale_gap = gap.sqrt() * migration_scale

    pyro.sample(
        "migration",
        dist.Normal(parent_location, migration_scale_gap.unsqueeze(-1)).to_event(1),
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
