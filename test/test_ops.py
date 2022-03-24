import math
from collections import namedtuple

import pytest
import torch
from tspyro.ops import CumlogsumexpUpTree
from tspyro.ops import CummaxUpTree
from tspyro.ops import CumsumUpTree

Edge = namedtuple("Edge", "child, parent")


class MockTS:
    def __init__(self, edges):
        self._edges = tuple(edges)

    def edges(self):
        return self._edges


OPS = {
    CumsumUpTree: torch.add,
    CummaxUpTree: torch.max,
    CumlogsumexpUpTree: torch.logaddexp,
}


def make_fake_ts(num_leaves: int, num_internal: int, branching_factor: int = 2):
    # Note the generated ts is directed but may not have a root.
    num_nodes = num_leaves + num_internal
    edges = []
    for p in range(num_leaves, num_nodes):
        for c in set(torch.randint(0, p, (branching_factor,)).tolist()):
            edges.append(Edge(c, p))
    return MockTS(edges)


@pytest.mark.parametrize(
    "num_leaves, num_internal, branching_factor",
    [
        (2, 1, 2),
        (10, 10, 2),
        (20, 20, 5),
        (100, 100, 10),
        (1000, 1000, 10),
    ],
    ids=str,
)
@pytest.mark.parametrize("Aggregate", [CumsumUpTree, CummaxUpTree, CumlogsumexpUpTree])
def test_aggregate_up_tree(num_leaves, num_internal, branching_factor, Aggregate):
    ts = make_fake_ts(num_leaves, num_internal, branching_factor)
    num_nodes = num_leaves + num_internal
    data = torch.rand(num_nodes).requires_grad_()

    # Run method under test.
    accumulate = Aggregate(ts)
    actual = accumulate(data)

    # Smoke test that gradients work.
    torch.autograd.grad(actual.sum(), [data])
    actual.detach_()
    data.detach_()

    # Check value. This computation works only because of our ordering.
    if Aggregate is CumsumUpTree:
        expected = data.detach().clone()
        for edge in ts.edges():
            expected[edge.parent] += expected[edge.child]
    else:
        op = OPS[Aggregate]
        expected = torch.zeros_like(data)
        expected[:num_leaves] = data[:num_leaves]
        expected[num_leaves:] = -math.inf
        pending = torch.zeros(num_nodes, dtype=torch.long)
        for _, p in ts.edges():
            pending[p] += 1
        for c, p in ts.edges():
            expected[p] = op(expected[p], expected[c])
            pending[p] -= 1
            if pending[p] == 0:  # all children have been aggregated
                expected[p] += data[p]
        assert pending.eq(0).all()
    assert torch.allclose(actual, expected, atol=1e-4)

    # Check inverse.
    actual_data = accumulate.inverse(actual)
    assert torch.allclose(actual_data, data, atol=1e-3)
