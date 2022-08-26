import torch
import numpy as np
from torch_scatter import scatter
from tspyro.ops import get_ancestral_geography
from torch import einsum
from torch.linalg import cholesky
import time


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
                 time_cutoff=50.0,
                 migration_scale=0.3,
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
        print("Using a time cutoff of {:.2f} with {} strategy".format(self.time_cutoff, self.strategy))

        # compute heuristic location estimates
        self.initial_loc = torch.zeros(self.num_nodes, 2, dtype=dtype, device=device)
        initial_loc = get_ancestral_geography(self.ts, self.locations[self.observed].data.cpu().numpy()).type_as(self.locations)
        self.initial_loc[self.unobserved] = initial_loc
        #assert self.initial_loc.isnan().sum().item() == 0.0

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

        self.migration_scale = torch.tensor(migration_scale, dtype=self.dtype, device=device)

        self.edge_times = (self.times[self.parent] - self.times[self.child]).clamp(min=1.0e-6)
        self.scaled_inv_edge_times = (self.edge_times * self.migration_scale.pow(2.0)).reciprocal()

        self.compute_b_lambda_diag()

    def compute_heuristic_metrics(self):
        mask = self.unobserved
        assert self.locations[mask].isnan().sum().item() == 0.0
        mrmse_heuristic = (self.locations[mask] - self.initial_loc[mask]).pow(2.0).sum(-1).sqrt().mean().item()
        rmse_heuristic = (self.locations[mask] - self.initial_loc[mask]).pow(2.0).sum(-1).mean().sqrt().item()
        print("rmse_heuristic:  {:.4f}".format(rmse_heuristic))
        print("mrmse_heuristic: {:.4f}".format(mrmse_heuristic))

    def compute_empirical_scale(self):
        parent_location = self.locations.index_select(-2, self.parent)  # num_edges 2
        child_location = self.locations.index_select(-2, self.child)
        delta_loc_sq = (parent_location - child_location).pow(2.0).mean(-1)  # num_edges
        mask = ~delta_loc_sq.isnan()
        scale = (delta_loc_sq / self.edge_times)[mask].sum().item() / mask.sum().item()
        print("empirical scale", scale)

    def do_cg(self, max_num_iter=500, tol=1.0e-5):
        t0 = time.time()

        x_prev = self.initial_loc
        r_prev = self.b - self.matmul(x_prev)
        p = r_prev
        #assert r_prev[self.observed].abs().max().item() == 0.0

        for i in range(max_num_iter):
            Ap = self.matmul(p)
            App = einsum("is,is->s", Ap, p)
            r_dot_r = einsum("is,is->s", r_prev, r_prev)
            r_dot_r_max = r_dot_r.max().item()
            alpha = r_dot_r / App
            x = x_prev + alpha * p
            r = r_prev - alpha * Ap
            delta_x = (x - x_prev).abs().max().item()
            if delta_x < tol:
                print("Terminating CG early at iteration {} with delta_x = {:.2e}".format(i, delta_x))
                break
            if i % 20 == 0:
                print("[CG step {:03d}]  r_dot_r_max: {:.2e}   delta_x: {:.2e}".format(i, r_dot_r_max, delta_x))
            beta = einsum("is,is->s", r, r) / r_dot_r
            p = r + beta * p
            x_prev, r_prev = x, r

        print("Time to do CG: {:.2f}".format(time.time() - t0))

        mask = self.unobserved
        mrmse = (x[mask] - self.locations[mask]).pow(2.0).sum(-1).sqrt().mean().item()
        rmse = (x[mask] - self.locations[mask]).pow(2.0).sum(-1).mean().sqrt().item()
        print("cg model rmse:   {:.4f}".format(rmse))
        print("cg model mrmse:  {:.4f}".format(mrmse))

        return x[mask]

    def test_b_lambda_diag(self):
        b = torch.zeros(self.num_nodes, 2, dtype=self.dtype, device=self.device)
        for parent_idx, child_loc, inv_edge_time in zip(self.single_edges_child_parent,
                                                        self.locations[self.single_edges_child_child],
                                                        self.scaled_inv_edge_times[self.singly_observed_child]):
            b[parent_idx] += child_loc * inv_edge_time
        for child_idx, parent_loc, inv_edge_time in zip(self.single_edges_parent_child,
                                                        self.locations[self.single_edges_parent_parent],
                                                        self.scaled_inv_edge_times[self.singly_observed_parent]):
            b[child_idx] += parent_loc * inv_edge_time

        b_delta = (self.b - b).abs().max().item()
        assert b_delta == 0.0
        assert self.b[self.observed].abs().max().item() == 0.0

        lambda_diag = torch.zeros(self.num_nodes, dtype=self.dtype, device=self.device)
        for parent_idx, child_idx, inv_edge_time in zip(self.double_edges_parent, self.double_edges_child,
                                                        self.scaled_inv_edge_times[self.doubly_unobserved]):
            lambda_diag[parent_idx] += inv_edge_time
            lambda_diag[child_idx] += inv_edge_time

        for parent_idx, inv_edge_time in zip(self.single_edges_child_parent,
                                             self.scaled_inv_edge_times[self.singly_observed_child]):
            lambda_diag[parent_idx] += inv_edge_time
        for child_idx, inv_edge_time in zip(self.single_edges_parent_child,
                                            self.scaled_inv_edge_times[self.singly_observed_parent]):
            lambda_diag[child_idx] += inv_edge_time

        lambda_diag_delta = (self.lambda_diag - lambda_diag).abs().max().item()
        assert lambda_diag_delta < 1.0e-10
        assert self.lambda_diag[self.observed].abs().max().item() == 0.0

        #print("lambda_diag[unobserved] min/max: {:.3f} / {:.3f}".format(
            #self.lambda_diag[self.unobserved].min().item(),
            #self.lambda_diag[self.unobserved].max().item()))

    def compute_b_lambda_diag(self):
        self.b = torch.zeros(self.num_nodes, 2, dtype=self.dtype, device=self.device)
        locations = self.locations  # torch.nan_to_num(self.locations, nan=0.0)

        src = locations[self.single_edges_child_child] * self.scaled_inv_edge_times[self.singly_observed_child, None]
        scatter(src=src, index=self.single_edges_child_parent, dim=-2, out=self.b)
        src = locations[self.single_edges_parent_parent] * self.scaled_inv_edge_times[self.singly_observed_parent, None]
        scatter(src=src, index=self.single_edges_parent_child, dim=-2, out=self.b)

        self.lambda_diag = torch.zeros(self.num_nodes, dtype=self.dtype, device=self.device)
        src = self.scaled_inv_edge_times[self.doubly_unobserved]
        scatter(src=src, index=self.double_edges_parent, dim=-1, out=self.lambda_diag)
        scatter(src=src, index=self.double_edges_child, dim=-1, out=self.lambda_diag)
        src = self.scaled_inv_edge_times[self.singly_observed_child]
        scatter(src=src, index=self.single_edges_child_parent, dim=-1, out=self.lambda_diag)
        src = self.scaled_inv_edge_times[self.singly_observed_parent]
        scatter(src=src, index=self.single_edges_parent_child, dim=-1, out=self.lambda_diag)

    def do_cholesky_inversion(self):
        lambda_prec = torch.zeros(self.num_nodes, self.num_nodes, dtype=self.dtype, device=self.device)
        for parent_idx, child_idx, inv_edge_time in zip(self.double_edges_parent, self.double_edges_child,
                                                        self.scaled_inv_edge_times[self.doubly_unobserved]):
            lambda_prec[parent_idx, parent_idx] += inv_edge_time
            lambda_prec[child_idx, child_idx] += inv_edge_time

        for parent_idx, child_idx, inv_edge_time in zip(self.double_edges_parent, self.double_edges_child,
                                                        self.scaled_inv_edge_times[self.doubly_unobserved]):
            lambda_prec[parent_idx, child_idx] -= inv_edge_time
            lambda_prec[child_idx, parent_idx] -= inv_edge_time

        for parent_idx, inv_edge_time in zip(self.single_edges_child_parent,
                                             self.scaled_inv_edge_times[self.singly_observed_child]):
            lambda_prec[parent_idx, parent_idx] += inv_edge_time
        for child_idx, inv_edge_time in zip(self.single_edges_parent_child,
                                            self.scaled_inv_edge_times[self.singly_observed_parent]):
            lambda_prec[child_idx, child_idx] += inv_edge_time

        t0 = time.time()
        mask = self.unobserved
        L = cholesky(lambda_prec[mask][:, mask], upper=False)
        b = self.b[mask]
        x = torch.cholesky_solve(b, L, upper=False)
        print("Time to do Cholesky: {:.2f}".format(time.time() - t0))

        mrmse = (x - self.locations[mask]).pow(2.0).sum(-1).sqrt().mean().item()
        rmse = (x - self.locations[mask]).pow(2.0).sum(-1).mean().sqrt().item()
        print("cholesky model rmse:  {:.4f}".format(rmse))
        print("cholesky model mrmse: {:.4f}".format(mrmse))

        return x

    def test_matmul(self, rhs):
        result = torch.zeros(self.num_nodes, 2, dtype=self.dtype, device=self.device)
        for parent_idx, child_idx, inv_edge_time in zip(self.double_edges_parent, self.double_edges_child,
                                                        self.scaled_inv_edge_times[self.doubly_unobserved]):
            result[parent_idx] -= inv_edge_time * rhs[child_idx]
            result[child_idx] -= inv_edge_time * rhs[parent_idx]

        matmul_result = self.matmul(rhs)
        assert matmul_result[self.observed].abs().max().item() == 0.0

        delta = self.lambda_diag[:, None] * rhs + result - matmul_result
        assert delta.abs().max().item() < 1.0e-4

    def matmul(self, rhs):
        assert rhs.shape == (self.num_nodes, 2)

        result = self.lambda_diag[:, None] * rhs
        delta_result = torch.zeros(self.num_nodes, 2, dtype=self.dtype, device=self.device)

        src = -self.scaled_inv_edge_times[self.doubly_unobserved][:, None] * rhs[self.double_edges_parent]
        scatter(src=src, index=self.double_edges_child, dim=-2, out=delta_result)
        src = -self.scaled_inv_edge_times[self.doubly_unobserved][:, None] * rhs[self.double_edges_child]
        scatter(src=src, index=self.double_edges_parent, dim=-2, out=delta_result)

        result += delta_result

        return result
