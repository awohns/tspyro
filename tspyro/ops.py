import logging
from abc import ABC
from abc import abstractmethod
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


class AccumulateUpTree(ABC):
    """
    Generic semiring programming up a tree.
    """

    def __init__(self, ts):
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
    def _aggregate_children(self, children, parent, data, dim) -> torch.Tensor:
        raise NotImplementedError


class CumsumUpTree(AccumulateUpTree):
    """
    Computes a cumulative sum up a tree, so each node gets the sum over its
    descendents including itself.  Note in time trees, if a descendent is
    reachable along multiple paths, it will be summed multiple times, once per
    path.
    """

    def _aggregate_children(self, child, parent, data, dim):
        """
        Compute a numerically stabile logsumexp by shifting by max.
        """
        child_data = data.index_select(dim, child)
        parent = parent.expand_as(child_data)
        return torch.zeros_like(data).scatter_add(dim, parent, child_data)


class CumlogsumexpUpTree(AccumulateUpTree):
    """
    Computes a cumulative logsumexp up a tree, so each node gets the sum over
    its descendents including itself.  Note in time trees, if a descendent is
    reachable along multiple paths, it will be summed multiple times, once per
    path.
    """

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
        exp_child_data = (child_data - max_child[..., parent]).exp()  # [E]
        logsumexp_child = (
            max_child
            + torch.zeros_like(max_child).scatter_add(dim, parent, exp_child_data).log()
        )  # [N]
        assert logsumexp_child.shape == data.shape
        return logsumexp_child
