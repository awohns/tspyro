import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


class NaiveModel(PyroModule):
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
        # timepoints = timepoints.clamp(min=0)
        grid_data = torch.as_tensor(prior.grid_data[:], dtype=torch.float)
        grid_data = grid_data / grid_data.sum(1, True)
        self.prior_loc = torch.einsum("t,nt->n", timepoints, grid_data)
        deltas = (timepoints - self.prior_loc[:, None]) ** 2
        self.prior_scale = torch.einsum("nt,nt->n", deltas, grid_data).sqrt()
        # self.prior_loc += math.log(Ne)
        # GEOGRAPHY
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
        # trying a different link function, exp in left tail, linear in right tail
        # internal_time = torch.nn.functional.softplus(unconstr_internal_time)
        # internal_time = torch.tensor(internal_time, dtype=torch.float)
        internal_time = internal_time  # * self.Ne
        time = torch.zeros((self.num_nodes,))
        time[self.is_internal] = internal_time
        # time = torch.tensor(sample_ts.tables.nodes.time)
        # GEOGRAPHY.
        # Should we be Bayesian about migration scale, or should it be fixed?
        migration_scale = pyro.sample("migration_scale", dist.LogNormal(-5, 1))

        # Next add a factor for time gaps between parents and children.
        gap = time[self.parent] - time[self.child]
        # print("MIG SCALE:", float(migration_scale), "GAP:", float(gap.mean()))
        with pyro.plate("edges", len(gap)):

            rate = (gap * self.span * self.mutation_rate).clamp(min=1e-8)
            pyro.sample(
                "mutations",
                dist.Poisson(rate),
                obs=self.mutations,
            )

            if self.leaf_location is not None:
                gap = gap.clamp(min=1e-10)
                # GEOGRAPHY.
                # The following encodes that children migrate away from their parents
                # following brownian motion with rate migration_scale.
                location = torch.cat([self.leaf_location, internal_location], 0)
                parent_location = location.index_select(0, self.parent)
                child_location = location.index_select(0, self.child)
                # Note we need to .unsqueeze(-1) i.e. [..., None] the migration_scale
                # in case you want to draw multiple samples.
                migration_radius = migration_scale[..., None] * gap ** 0.5

                # Normalise distance
                distance = (child_location - parent_location).square().sum(-1).sqrt()
                pyro.sample(
                    "migration",
                    # Trying a heavy-tailed distribution
                    dist.Exponential(1 / migration_radius),
                    obs=distance
                    # dist.Normal(torch.zeros_like(parent_location),
                    # migration_radius.unsqueeze(-1)).to_event(1),
                    # obs=child_location - parent_location,
                )
            else:
                location = torch.ones(self.ts.num_nodes)
        return time, gap, location, migration_scale
