import torch
import numpy as np
from tspyro.ops import get_mut_edges
from torch_scatter import scatter


def get_metadata(ts):
    locations = []
    for node in ts.nodes():
        if node.individual != -1:
            loc = ts.individual(node.individual).location[:2]
            if len(loc) == 2:
                locations.append(ts.individual(node.individual).location[:2])
            else:
                locations.append(np.array([np.nan, np.nan]))
        else:
            locations.append(np.array([np.nan, np.nan]))

    is_leaf = np.array(ts.tables.nodes.flags & 1, dtype=bool)
    is_internal = ~is_leaf
    true_times = ts.tables.nodes.time

    return np.array(locations), true_times, is_leaf, is_internal


class CG(object):
    def __init__(self,
                 ts,
                 migration_scale=0.3,
                 device=torch.device('cpu')):

        super().__init__()
        nodes = ts.tables.nodes
        edges = ts.tables.edges
        self.ts = ts
        self.device = device

        self.observed = torch.tensor((nodes.flags & 1).astype(bool), dtype=torch.bool, device=device)
        self.unobserved = ~self.observed
        self.num_nodes = len(self.observed)
        self.num_unobserved = int(self.unobserved.sum().item())
        self.num_observed = int(self.observed.sum().item())
        print("num_unobserved", self.num_unobserved, "num_observed", self.num_observed)
        assert self.num_nodes == self.num_unobserved + self.num_observed

        self.parent = torch.tensor(edges.parent, dtype=torch.long, device=device)
        self.child = torch.tensor(edges.child, dtype=torch.long, device=device)
        self.span = torch.tensor(edges.right - edges.left, dtype=torch.float, device=device)
        self.mutations = torch.tensor(get_mut_edges(self.ts), dtype=torch.float, device=device)

        self.parent_observed = self.observed[self.parent]
        self.child_observed = self.observed[self.child]
        self.parent_unobserved = self.unobserved[self.parent]
        self.child_unobserved = self.unobserved[self.child]
        assert self.parent_observed.max().item() == 0.0

        self.doubly_unobserved = self.parent_unobserved & self.child_unobserved
        self.singly_observed = self.parent_unobserved & self.child_observed
        print("doubly_unobserved", self.doubly_unobserved.sum().item(),
              "singly_observed", self.singly_observed.sum().item())
        assert self.doubly_unobserved.sum().item() + self.singly_observed.sum().item() == self.parent.size(0)

        self.double_edges_parent = self.parent[self.doubly_unobserved]
        self.double_edges_child = self.child[self.doubly_unobserved]
        self.single_edges_parent = self.parent[self.singly_observed]
        self.single_edges_child = self.child[self.singly_observed]
        print("double_edges_parent", self.double_edges_parent.shape,
              "single_edges_parent", self.single_edges_parent.shape)

        self.migration_scale = torch.tensor(migration_scale, dtype=torch.float, device=device)

        self.times = torch.as_tensor(ts.tables.nodes.time, dtype=torch.float, device=device)
        self.edge_times = (self.times[self.parent] - self.times[self.child]).clamp(min=1.0)
        self.scaled_inv_edge_times = (self.edge_times * self.migration_scale.pow(2.0)).reciprocal()

        self.locations = torch.as_tensor(get_metadata(self.ts)[0], dtype=torch.float, device=device)
        print("locations.shape", self.locations.shape)

        self.prep_matrix()

    def prep_matrix(self):
        self.b = torch.zeros(self.num_unobserved, 2, dtype=torch.float, device=self.device)
        locations = torch.nan_to_num(self.locations, nan=0.0)


        src = locations[self.single_edges_child] * self.scaled_inv_edge_times[self.singly_observed, None]
        scatter(src=src, index=self.single_edges_parent, dim=-2, out=self.b)

        b2 = torch.zeros(self.num_unobserved, 2, dtype=torch.float, device=self.device)
        for parent_idx, child_loc, inv_edge_time in zip(self.single_edges_parent,
                                                        locations[self.single_edges_child],
                                                        self.scaled_inv_edge_times[self.singly_observed]):
            b2[parent_idx] += child_loc * inv_edge_time

        b_delta = (self.b - b2).abs().max().item()
        assert b_delta == 0.0

        return

        self.lambda_diag = torch.zeros(self.num_nodes, dtype=torch.float, device=self.device)
        src = self.child_unobserved.float() * self.parent_unobserved.float() * self.scaled_inv_edge_times
        scatter(src=src, index=self.parent, dim=-1, out=self.lambda_diag)
        scatter(src=src, index=self.child, dim=-1, out=self.lambda_diag)

        lambda_diag2 = torch.zeros(self.num_nodes, dtype=torch.float, device=self.device)
        for parent_idx, child_idx, inv_edge_time, parent_unobs, child_unobs in \
            zip(self.parent, self.child, self.scaled_inv_edge_times, self.parent_unobserved, self.child_unobserved):
                if child_unobs and parent_unobs:
                    lambda_diag2[parent_idx] += inv_edge_time
                    lambda_diag2[child_idx] += inv_edge_time

        lambda_diag_delta = (self.lambda_diag - lambda_diag2).abs().max().item()
        assert lambda_diag_delta == 0.0

        #self.b = self.b[self.unobserved]
        #self.lambda_diag = self.lambda_diag[self.unobserved]

        #assert self.b.shape == (self.num_unobserved, 2)
        #assert self.lambda_diag.shape == (self.num_unobserved,)

        self.matmul(self.b)

    def matmul(self, rhs):
        assert rhs.shape == (self.num_nodes, 2)

        result = self.lambda_diag[:, None] * rhs

        delta_result = torch.zeros(self.num_nodes, 2, dtype=torch.float, device=self.device)
        src = -self.child_unobserved.float() * self.parent_unobserved.float() * self.scaled_inv_edge_times
        src = rhs[self.parent] * src[:, None]
        scatter(src=src, index=self.parent, dim=-2, out=delta_result)

        delta_result2 = torch.zeros(self.num_nodes, 2, dtype=torch.float, device=self.device)
        for parent_idx, child_idx, inv_edge_time, parent_unobs, child_unobs in \
            zip(self.parent, self.child, self.scaled_inv_edge_times, self.parent_unobserved, self.child_unobserved):
                if child_unobs and parent_unobs:
                    delta_result2[parent_idx] -= inv_edge_time * rhs[parent_idx]

        delta_delta = (delta_result - delta_result2).abs().max().item()
        assert delta_delta == 0.0

        #assert result.shape == (self.num_unobserved, 2)
