import torch
import numpy as np
from tspyro.ops import get_mut_edges


class CG(object):
    def __init__(self, ts, device=torch.device('cpu')):
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
        assert self.num_nodes == self.num_unobserved + self.num_observed

        self.parent = torch.tensor(edges.parent, dtype=torch.long, device=device)
        self.child = torch.tensor(edges.child, dtype=torch.long, device=device)
        self.span = torch.tensor(edges.right - edges.left, dtype=torch.float, device=device)
        self.mutations = torch.tensor(get_mut_edges(self.ts), dtype=torch.float, device=device)
