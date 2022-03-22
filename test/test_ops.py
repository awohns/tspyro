from collections import namedtuple

import pytest
import torch
from tspyro.ops import CumsumUpTree

Edge = namedtuple("Edge", "child, parent")


class MockTS:
    def __init__(self, edges):
        self._edges = tuple(edges)

    def edges(self):
        return self._edges


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
def test_cumsum_up_tree(num_leaves, num_internal, branching_factor):
    ts = make_fake_ts(num_leaves, num_internal, branching_factor)
    num_nodes = num_leaves + num_internal
    data = torch.rand(num_nodes).requires_grad_()

    # Run method under test.
    cumsum = CumsumUpTree(ts)
    actual = cumsum(data)

    # Smoke test that gradients work.
    torch.autograd.grad(actual.sum(), [data])
    actual.detach_()

    # Check value. This computation works only because of our ordering.
    expected = data.detach().clone()
    for edge in ts.edges():
        expected[edge.parent] += expected[edge.child]
    assert torch.allclose(expected, actual, atol=1e-4)
