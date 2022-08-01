from abc import abstractmethod
import typing
import typing_extensions

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import tskit

from pyro import poutine
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroSample, pyro_method

import tspyro.diffusion as diffusion
import tspyro.ops as ops


class TSPyroModule(PyroModule):
    """
    Convenient Pyro module that holds an updatable tree sequence.
    """
    def __init__(self, ts: tskit.TreeSequence):
        super().__init__()
        self.ts = ts

    @property
    def ts(self) -> tskit.TreeSequence:
        return self._ts

    @ts.setter
    def ts(self, ts: tskit.TreeSequence):
        self._ts = ts

    @property
    def nodes(self):
        return self.ts.tables.nodes

    @property
    def edges(self):
        return self.ts.tables.edges

    @property
    def is_leaf(self):
        return torch.tensor((nodes.flags & 1).astype(bool), dtype=torch.bool)

    @property
    def is_internal(self):
        return ~self.is_leaf

    @property
    def num_nodes(self):
        return len(self.is_leaf)

    @property
    def num_internal(self):
        return self.is_internal.sum().item()

    @property
    def parent(self):
        return torch.tensor(self.edges.parent, dtype=torch.long)

    @property
    def child(self):
        return torch.tensor(self.edges.child, dtype=torch.long)

    @property
    def span(self):
        return torch.tensor(self.edges.right - self.edges.left)


class TSNodeModel(TSPyroModule):
    """
    Base class for models of individual variables on tree sequences.
    """

    @abstractmethod
    def initialize(self, site: typing.Dict[str, typing.Any]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def global_model(self, *args, **kwargs) -> typing.Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def internal_model(self, *args, **kwargs) -> typing.Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def leaf_model(self, internal_values, *args, **kwargs) -> typing.Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def edge_model(self, node_values, *args, **kwargs) -> typing.Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError("shouldn't be here!")


class GenericBaseModel(TSPyroModule):
    """
    Joint generative models of tree sequence variables.
    """
    def __init__(self, ts: tskit.TreeSequence, *models: typing.Optional[TSNodeModel]):
        super().__init__(ts)
        self.models = PyroModule[torch.nn.ModuleList](*models)

    def forward(self, *args, **kwargs):
        internal_nodes_plate = pyro.plate("internal_nodes", self.num_internal)
        leaf_nodes_plate = pyro.plate("leaf_nodes", self.num_nodes - self.num_internal)
        edges_plate = pyro.plate("edges", self.num_edges)

        global_values = {}
        for model in self.models:
            global_values.update(model.global_model())

        internal_values = {}
        with internal_nodes_plate:
            for model in self.models:
                internal_values.update(model.internal_model(global_values))

        leaf_values = {}
        with leaf_nodes_plate:
            for model in self.models:
                leaf_values.update(model.leaf_model(internal_values))

        node_values = ...

        with edges_plate:
            for model in self.models:
                model.edge_model(node_values)

        return node_values


class TimeModel(TSNodeModel):

    def __init__(
        self,
        ts: tskit.TreeSequence,
        mutation_rate: typing.Union[float, torch.Tensor]
    ):
        super().__init__(ts)
        self.mutation_rate = mutation_rate

    @PyroSample
    def internal_time(self):
        return dist.LogNormal(self.prior_loc, self.prior_scale)

    @pyro_method
    def internal_model(self):
        internal_time = self.internal_time  # * self.Ne
        return internal_time

    @pyro_method
    def leaf_model(self, internal_values):
        internal_time = ...
        time = torch.zeros(internal_time.shape[:-1] + (self.num_nodes,))
        time[..., self.is_internal] = internal_time
        return time

    @pyro_method
    def edge_model(self, node_values):
        # Next add a factor for time gaps between parents and children.
        time = ...
        gap = time[..., parent] - time[..., child]

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


class GenotypeModel(TSNodeModel):

    def __init__(self, ts: tskit.TreeSequence, *args):
        super().__init__(ts)

    @pyro_method
    def internal_model(self, global_values):
        ...  # TODO ancestral haplotypes

    @pyro_method
    def leaf_model(self, internal_values):
        ...  # TODO impute missing sample data

    @pyro_method
    def edge_model(self, node_values):
        ...  # TODO


class BaseLocationModel(TSNodeModel):

    def __init__(self, ts: tskit.TreeSequence, *args):
        super().__init__(ts)

    def internal_model(self):
        # Sample location from a flat prior, we'll add a pyro.factor statement later.
        location = pyro.sample(
            "internal_location",
            dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1).mask(False),
        )
        return location
  
    
class EuclideanLocationModel(BaseLocationModel):

    def __init__(self, ts: tskit.TreeSequence, *args):
        super().__init__(ts)
        ...

    @PyroSample
    def migration_scale(self):
        return dist.LogNormal(0, 4)

    @pyro_method
    def global_model(self):
        migration_scale = self.migration_scale
        return migration_scale

    @pyro_method
    def leaf_model(self, internal_values):
        ...

    @pyro_method
    def edge_model(self, node_values):
        # The following encodes that children migrate away from their parents
        # following approximately Brownian motion with rate migration_scale.
        migration_scale = ...
        location, gap = ...

        parent_location = location.index_select(-2, self.parent)
        child_location = location.index_select(-2, self.child)

        # Note we need to .unsqueeze(-1) i.e. [..., None] the migration_scale
        # in case you want to draw multiple samples.
        migration_radius = migration_scale[..., None] * gap ** 0.5

        distance = torch.linalg.norm(child_location - parent_location, dim=-1, ord=2)
        distance = distance.clamp(min=1e-6)
        pyro.sample(
            "migration",
            dist.Gamma(2, 1 / migration_radius),
            obs=distance,
        )


class SphericalEuclideanLocationModel(BaseLocationModel):

    @pyro_method
    def internal_model(self):
        internal_location_sphere = pyro.sample(
            "internal_location_sphere",
            dist.VonMises(...)
        )
        internal_location = pyro.deterministic(
            "internal_location", xyz_to_latlong(internal_location_sphere))  # TODO
        return internal_location

    @pyro_method
    def edge_model(self, node_values):

        location, gap = ...
        migration_scale = ...

        # The following encodes that children migrate away from their parents
        # following approximately Brownian motion with rate migration_scale.
        parent_location = location.index_select(-2, self.parent)
        child_location = location.index_select(-2, self.child)

        # Note we need to .unsqueeze(-1) i.e. [..., None] the migration_scale
        # in case you want to draw multiple samples.
        migration_radius = migration_scale[..., None] * gap ** 0.5

        child_xyz = ops.latlong_to_xyz(child_location)
        parent_xyz = ops.latlong_to_xyz(parent_location)
        distance = torch.linalg.norm(child_xyz - parent_xyz, dim=-1, ord=2)
        pyro.sample(
            "migration",
            dist.Gamma(2, 1 / migration_radius),
            obs=distance,
        )


class WayPointLocationModel(BaseLocationModel):

    def __init__(
        self,
        ts: tskit.TreeSequence,
        transitions: torch.Tensor,
        waypoints: torch.Tensor,
        waypoint_radius: typing.Union[float, torch.Tensor]
    ):
        super().__init__(ts)  # TODO
        self.waypoint_radius = waypoint_radius
        self.waypoints = waypoints
        # This cheaply precomputes some matrices.
        self.matrix_exp = diffusion.ApproximateMatrixExponential(
            transitions, max_time_step=1e6
        )  # TODO: need to fix max time step

    @pyro_method
    def edge_model(self, node_values):
        location, gap = ...
        migration_scale = ...

        parent_location = location.index_select(0, self.parent)
        child_location = location.index_select(0, self.child)
        pyro.sample(
            "migration",
            diffusion.WaypointDiffusion2D(
                source=parent_location,
                time=gap * migration_scale,
                radius=self.waypoint_radius,
                waypoints=self.waypoints,
                matrix_exp=self.matrix_exp,
            ),
            obs=child_location,
        )
