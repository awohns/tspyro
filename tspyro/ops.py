import logging
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


class CumsumUpTree:
    """
    Computes a cumulative sum up a tree, so each node gets the sum over its
    descendents including itself.  Note in time trees, if a descendent is
    reachable along multiple paths, it will be summed multiple times, once per
    path.
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
            stage = torch.tensor(
                [
                    [child, parent]
                    for parent in parents
                    for child in parent_to_children[parent]
                ]
            ).T.contiguous()
            self.stages.append(stage)

        # Log computational complexity.
        stage_sizes = [stage.size(-1) for stage in self.stages]
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
        result = data
        for child, parent in self.stages:
            child_result = result.index_select(dim, child)
            parent = parent.expand_as(child_result)
            result = result.scatter_add(dim, parent, child_result)
        return result

    def inverse(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim >= 0:
            dim -= data.dim()
        if dim != -1:
            raise NotImplementedError(f"TODO support dim={dim}")

        # Apply stages in succession.
        result = data
        for child, parent in reversed(self.stages):
            child_result = result.index_select(dim, child).neg()
            parent = parent.expand_as(child_result)
            result = result.scatter_add(dim, parent, child_result)
        return result
