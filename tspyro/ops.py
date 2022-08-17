import itertools
import logging
import operator
import typing

from abc import ABC
from abc import abstractmethod
from collections import defaultdict

import torch
import tskit

import numpy as np

logger = logging.getLogger(__name__)


class AccumulateUpTree(ABC):
    """
    Generic semiring programming up a tree.
    """

    def __init__(self, ts: tskit.TreeSequence):
        super().__init__()
        # Compute rank of each node.
        num_nodes = 1 + max(max(e.child, e.parent) for e in ts.edges())
        ranks = [0] * num_nodes
        changed = True
        while changed:
            changed = False
            for edge in ts.edges():
                new_rank = ranks[edge.child] + 1
                old_rank = ranks[edge.parent]
                if old_rank < new_rank:
                    ranks[edge.parent] = new_rank
                    changed = True

        # Group parents by rank.
        rank_to_parents = defaultdict(list)
        for n, rank in enumerate(ranks):
            if rank > 0:
                rank_to_parents[rank].append(n)

        # Group children by parent.
        parent_to_children = defaultdict(set)
        for edge in ts.edges():
            parent_to_children[edge.parent].add(edge.child)

        # Compute a series of sparse tensors, one per rank.
        self.stages = []
        for _, parents in sorted(rank_to_parents.items()):
            child_parent = torch.tensor(
                [
                    [child, parent]
                    for parent in parents
                    for child in parent_to_children[parent]
                ]
            ).T.contiguous()
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[parents] = True
            self.stages.append((child_parent, mask))

        # Log computational complexity.
        stage_sizes = [stage[0].size(-1) for stage in self.stages]
        num_edges = sum(stage_sizes)
        logger.info(
            "Created a CumsumUpTree "
            f"with {num_edges} edges in {len(self.stages)} stages of size:\n"
            f"{stage_sizes}"
        )

    def __call__(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:

        if dim >= 0:
            dim -= data.dim()
        if dim != -1:
            raise NotImplementedError(f"TODO support dim={dim}")

        # Apply stages in succession.
        result = data.clone()
        for (child, parent), mask in self.stages:
            diff = self._aggregate_children(child, parent, result, dim)
            mask = mask.expand_as(data)
            result[mask] += diff[mask]
        return result

    def inverse(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim >= 0:
            dim -= data.dim()
        if dim != -1:
            raise NotImplementedError(f"TODO support dim={dim}")

        # Apply stages in succession.
        result = data.clone()
        for (child, parent), mask in reversed(self.stages):
            diff = self._aggregate_children(child, parent, result, dim)
            mask = mask.expand_as(data)
            result[mask] -= diff[mask]
        return result

    @abstractmethod
    def _aggregate_children(
        self,
        child: torch.Tensor,
        parent: torch.Tensor,
        data: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
        raise NotImplementedError


class CumsumUpTree(AccumulateUpTree):
    """
    Computes a cumulative sum up a tree, so each node gets the sum over its
    descendents including itself.  Note in time trees, if a descendent is
    reachable along multiple paths, it will be summed multiple times, once per
    path.
    """

    def _aggregate_children(self, child, parent, data, dim):
        child_data = data.index_select(dim, child)
        parent = parent.expand_as(child_data)
        return torch.zeros_like(data).scatter_add(dim, parent, child_data)


class CummaxUpTree(AccumulateUpTree):
    """
    Computes a cumulative max up a tree, so each node gets the max over its
    descendents plus its own contribution. Note in time trees, if a descendent
    is reachable along multiple paths, it will be summed multiple times, once
    per path.
    """

    def _aggregate_children(self, child, parent, data, dim):
        from torch_scatter import scatter

        if dim != -1:
            raise NotImplementedError(f"TODO support dim={dim}")

        child_data = data.index_select(dim, child)  # [E]
        parent = parent.expand_as(child_data)  # [E]
        return scatter(
            child_data, parent, dim=dim, reduce="max", dim_size=data.shape[dim]
        )


class CumlogsumexpUpTree(AccumulateUpTree):
    """
    Computes a cumulative logsumexp up a tree, so each node gets the logsumexp
    over its descendents plus its own contribution.  Note in time trees, if a
    descendent is reachable along multiple paths, it will be summed multiple
    times, once per path.
    """
    def __init__(self, ts: tskit.TreeSequence, scale_factor=1.0):
        super().__init__(ts=ts)
        self.scale_factor = scale_factor

    def _aggregate_children(self, child, parent, data, dim):
        """
        Compute a numerically stabile logsumexp by shifting by max.
        """
        from torch_scatter import scatter

        if dim != -1:
            raise NotImplementedError(f"TODO support dim={dim}")

        child_data = data.index_select(dim, child)  # [E]
        parent = parent.expand_as(child_data)  # [E]
        max_child = scatter(
            child_data, parent, dim=dim, reduce="max", dim_size=data.shape[dim]
        )  # [N]
        exp_child_data = (self.scale_factor * (child_data - max_child[..., parent])).exp()  # [E]
        logsumexp_child = (
            max_child
            + torch.zeros_like(max_child).scatter_add(dim, parent, exp_child_data).log()
        )  # [N]
        assert logsumexp_child.shape == data.shape
        return logsumexp_child / self.scale_factor


def radians_center_weighted(
    x: typing.Sequence[float],
    y: typing.Sequence[float],
    z: typing.Sequence[float],
    weights: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
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


def weighted_geographic_center(
    lat_list: typing.List[float],
    long_list: typing.List[float],
    weights: typing.Iterable
) -> typing.Tuple[np.ndarray, np.ndarray]:
    x = list()
    y = list()
    z = list()
    if len(lat_list) == 1 and len(long_list) == 1:
        return (np.array(lat_list[0]), np.array(long_list[0]))
    lat_radians = np.radians(lat_list)
    long_radians = np.radians(long_list)
    x = np.cos(lat_radians) * np.cos(long_radians)
    y = np.cos(lat_radians) * np.sin(long_radians)
    z = np.sin(lat_radians)
    weights = np.array(weights)
    central_latitude, central_longitude = radians_center_weighted(
        x, y, z, weights
    )
    return (np.degrees(central_latitude), np.degrees(central_longitude))


def edges_by_parent_asc(
    ts: tskit.TreeSequence
) -> typing.Iterable[typing.Tuple[int, typing.Iterable[tskit.Edge]]]:
    """
    Return an itertools.groupby object of edges grouped by parent in ascending order
    of the time of the parent. Since tree sequence properties guarantee that edges
    are listed in nondecreasing order of parent time
    (https://tskit.readthedocs.io/en/latest/data-model.html#edge-requirements)
    we can simply use the standard edge order
    """
    return itertools.groupby(ts.edges(), operator.attrgetter("parent"))


def average_edges(
    parent_edges: typing.Tuple[int, typing.Iterable[tskit.Edge]],
    locations: np.ndarray,
    method="average"
) -> typing.Tuple[int, np.ndarray]:
    parent = parent_edges[0]
    edges = parent_edges[1]

    child_spanfracs = list()
    child_lats = list()
    child_longs = list()

    for edge in edges:
        child_spanfracs.append(edge.span)
        child_lats.append(locations[edge.child][0])
        child_longs.append(locations[edge.child][1])
    if method == "average":
        val = np.average(np.array([child_lats, child_longs]).T, axis=0)
    elif method == "geographic_mean":
        val = weighted_geographic_center(
            child_lats, child_longs, np.ones_like(len(child_lats))
        )
    return parent, val


def get_mut_edges(ts: tskit.TreeSequence) -> np.ndarray:
    """
    Get the number of mutations on each edge in the tree sequence.
    """
    edge_diff_iter = ts.edge_diffs()
    right = 0
    edges_by_child: typing.Dict[int, int] = {}  # contains {child_node:edge_id}
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


def get_ancestral_geography(
    ts: tskit.TreeSequence,
    sample_locations: np.ndarray,
    show_progress: typing.Optional[bool] = False
) -> torch.Tensor:
    """
    Use dynamic programming to find approximate posterior to sample from
    """
    locations = np.zeros((ts.num_nodes, 2))
    locations[ts.samples()] = sample_locations
    fixed_nodes = set(ts.samples())
    is_internal = ~torch.tensor((ts.tables.nodes.flags & 1).astype(bool), dtype=torch.bool)

    # Iterate through the nodes via groupby on parent node
    for parent_edges in edges_by_parent_asc(ts):
        if parent_edges[0] not in fixed_nodes:
            parent, val = average_edges(parent_edges, locations)
            locations[parent] = val
    return torch.tensor(
        locations[is_internal], dtype=torch.get_default_dtype()  # noqa: E203
    )


def great_circle(
    lon1: torch.Tensor,
    lat1: torch.Tensor,
    lon2: torch.Tensor,
    lat2: torch.Tensor
) -> torch.Tensor:
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


def great_secant(
    lon1: torch.Tensor,
    lat1: torch.Tensor,
    lon2: torch.Tensor,
    lat2: torch.Tensor
) -> torch.Tensor:
    lon = torch.deg2rad(torch.stack([lon1, lon2]))
    lat = torch.deg2rad(torch.stack([lat1, lat2]))
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    r2 = (x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2 + (z[0] - z[1]) ** 2
    return torch.sqrt(r2.clamp(min=1e-10))


def latlong_to_xyz(latlong: torch.Tensor) -> torch.Tensor:
    lat, long = latlong.unbind(-1)
    x = lat.cos() * long.cos()
    y = lat.cos() * long.sin()
    z = lat.sin()
    return torch.cat([x, y, z], dim=-1)
