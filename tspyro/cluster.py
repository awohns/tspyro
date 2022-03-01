import collections
import math

import networkx as nx
import numpy as np
import torch
import tqdm
from community import community_louvain


def check_sparse_genotypes(data: dict):
    """
    Check that a sparse genotype datastructure has correct fields, shapes, and
    dtypes.

    :param dict data: A dict representing sparse genotypes.
    :returns: A tuple ``(offsets, index, values)``
    :rtype: tuple
    """
    assert isinstance(data, dict)
    assert set(data) == {"offsets", "index", "values"}
    offsets = data["offsets"]
    index = data["index"]
    values = data["values"]

    assert isinstance(offsets, torch.Tensor)
    assert isinstance(index, torch.Tensor)
    assert isinstance(values, torch.Tensor)

    assert offsets.dtype == torch.long
    assert index.dtype == torch.long
    assert values.dtype == torch.bool

    assert offsets.dim() == 2
    assert index.dim() == 1
    assert values.dim() == 1

    assert index.shape == (int(offsets[-1, -1]),)
    assert index.shape == values.shape

    return offsets, index, values


def make_fake_data(num_samples, num_variants):
    """
    Make fake sparse genotype data (for testing and benchmarking).

    :returns: A dict representing sparse genotypes.
    :rtype: dict
    """
    dense_values = torch.full((num_samples, num_variants), 0.5).bernoulli_().bool()
    dense_mask = torch.full((num_samples, num_variants), 0.5).bernoulli_().bool()

    offsets = []
    index = []
    values = []

    end = 0
    for data, mask in zip(dense_values, dense_mask):
        index.append(mask.nonzero(as_tuple=True)[0])
        values.append(data[mask])
        beg, end = end, end + len(index[-1])
        offsets.append([beg, end])

    offsets = torch.tensor(offsets)
    index = torch.cat(index)
    values = torch.cat(values)
    data = dict(offsets=offsets, index=index, values=values)
    check_sparse_genotypes(data)

    return data


def naive_encoder(ts):
    """
    Make an encoding of the sparse genotype data naively: haplotype by haplotype.
    Note this is very slow for large tree sequences.

    :returns: A Dict representing sparse genotypes.
    :rtype: dict
    """
    offsets = []
    index = []
    values = []
    for haplo in tqdm.tqdm(ts.haplotypes(), total=ts.num_samples):
        begin = len(index)
        for i, g in enumerate(haplo):
            assert g in "-01"
            if g != "-":
                index.append(i)
                values.append(bool(int(g)))
        end = len(index)
        offsets.append([begin, end])
    return {
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "index": torch.tensor(index, dtype=torch.long),
        "values": torch.tensor(values, dtype=torch.bool),
    }


def variant_encoder(ts):
    """
    Encodes sparse genotype data variant by variant. This is faster
    than the naive encoding but requires inverting to be compatible with
    `make_clustering_gibbs()`.

    :returns: A Dict representing sparse genotypes in compressed sparse
    column format.
    :rtype: dict
    """
    offsets = []
    index = []
    values = []
    for var in tqdm.tqdm(ts.variants(), total=ts.num_sites):
        geno = var.genotypes
        begin = len(index)

        index.extend(np.where([geno != -1])[1])

        non_missing = geno != -1
        values.extend(geno[non_missing] != 0)
        end = len(index)
        offsets.append([begin, end])
    return {
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "index": torch.tensor(index, dtype=torch.long),
        "values": torch.tensor(values, dtype=torch.bool),
    }


def dense_encoder_genos(genos):
    """
    Encodes a genotype matrix in the sparse genotype encoding format.

    :returns: A Dict representing sparse genotypes.
    :rtype: dict
    """
    n, p = genos.shape
    print("n*p is {}".format(n * p))
    return {
        "offsets": torch.arange(n, dtype=torch.long)[:, None] * p
        + torch.tensor([0, p]),
        "index": torch.arange(p).expand(n, p).reshape(-1),
        "values": torch.as_tensor(genos).bool().reshape(-1),
    }


def make_clustering_gibbs(
    data: dict,
    num_clusters: int,
    *,
    num_epochs: int = 10,
) -> torch.Tensor:
    """
    Clusters sparse genotypes using subsample-annealed Gibbs sampling.

    :param dict data: A dict representing sparse genotypes.
    :param int num_clusters: The desired number of clusters.
    :param num_epochs: An optional number of epochs. Must be at least 1.
    :returns: A ``(num_variants,num_clusters)``-shaped tensor of clusters.
    :rtype: torch.Tensor
    """
    assert num_epochs >= 1
    offsets, index, values = check_sparse_genotypes(data)
    N = len(offsets)
    P = 1 + int(index.max())
    C = num_clusters
    MISSING = -1
    prior = 0.5  # Jeffreys prior

    # The single site Gibbs algorithm treats assignment as the latent variable,
    # and caches sufficient statistics in the counts tensor.
    assignment = torch.full((N,), MISSING, dtype=torch.long)
    counts = torch.full((P, C, 2), prior, dtype=torch.float)
    assert N < 2 ** 23, "counts has too little precision"

    # Use a shuffled linearly annealed schedule.
    schedule = (([+1, -1] * num_epochs)[:-1]) * N  # +1 means add, -1 means remove.
    shuffle = torch.randperm(N)
    assert sum(schedule) == N
    pending = {+1: 0, -1: 0}
    for sign in tqdm.tqdm(schedule):
        # Select the next pending datum to either add or remove.
        n = shuffle[pending[sign] % N]
        pending[sign] += 1
        beg, end = offsets[n]
        index_n = index[beg:end]
        value_n = values[beg:end]

        if sign > 0:
            # Add the datum to a random cluster.
            assert assignment[n] == MISSING
            posterior = counts[index_n]
            tails, heads = posterior.unbind(-1)
            logits = torch.where(value_n[:, None], heads, tails)
            logits = logits.div_(posterior.sum(-1)).log_().sum(0)
            logits -= logits.max()
            c = int(logits.exp_().multinomial(1))
            assignment[n] = c
        else:
            # Remove the datum from the current cluster.
            assert assignment[n] != MISSING
            c = int(assignment[n])
            assignment[n] = MISSING

        counts[index_n, c, value_n.long()] += sign
    assert all(assignment >= 0)
    assert pending[+1] % N == 0
    assert pending[-1] % N == 0
    expected = len(values) + counts.shape.numel() * prior
    assert (counts.sum() / expected).sub(1).abs() < 1e-6

    clusters = (counts[..., 1] / counts.sum(-1)).round_().bool()
    return assignment, clusters


# Functions borrowed from https://github.com/tskit-dev/what-is-an-arg-paper


def convert_nx(ts, mutation_rate=1e-8):
    """
    Returns the specified tree sequence as an networkx directed graph.
    Weights are the edge's total span * e^(-edge length * mutation_rate).

    Mutation rate is in units: probability of mutation per base pair per generation.
    """
    G = nx.Graph()
    edges = collections.defaultdict(list)
    times = ts.tables.nodes.time
    for edge in ts.edges():
        edges[(edge.child, edge.parent)].append(
            (edge.left, edge.right, times[edge.parent], times[edge.child])
        )
    for node in ts.nodes():
        G.add_node(node.id, time=node.time, flags=node.flags)
    for edge, intervals in edges.items():
        G.add_edge(
            *edge,
            weight=sum(
                (right - left) * math.e ** (-(top - bottom) * mutation_rate)
                for (left, right, top, bottom) in intervals
            ),
        )

    return G


def make_clustering_louvain(ts, resolution, max_cluster_size):
    """
    Clusters a tree sequence using the Louvain community detection algorithm.

    :param tskit.TreeSequence ts: The input tree sequence
    :param float resolution: Passed to community_louvain: "Will change the size of
        the communities, default to 1. represents the time described in
        “Laplacian Dynamics and Multiscale Modular Structure in Networks”, R.
        Lambiotte, J.-C. Delvenne, M. Barahona"
    :param int max_cluster_size: Maximum cluster size to calculate centroids from,
        usually 10000 nodes or less to avoid memory issues.
    :returns: A ``(num_variants,num_clusters)``-shaped tensor of clusters centroids.
    :rtype: torch.Tensor
    """
    graph = convert_nx(ts)

    partition = community_louvain.best_partition(
        graph, weight="weight", resolution=0.6
    )  # resolution > 1: fewer clusters. < 1: more clusters
    assignment = torch.tensor(list(partition.values()))
    clusters = torch.unique(assignment)
    cluster_centroids = np.zeros((len(clusters), ts.num_sites))
    for cluster in tqdm.tqdm(clusters[:]):
        cluster_assignment = torch.tensor(assignment)[assignment == cluster]
        if sum(assignment == cluster) > max_cluster_size:
            cluster_assignment = torch.randperm(len(cluster_assignment))[
                :max_cluster_size
            ]
        tables = ts.dump_tables()
        arr = np.zeros_like(tables.nodes.flags)
        arr[cluster_assignment] = np.ones_like(cluster_assignment)
        tables.nodes.flags = arr
        all_samples = tables.tree_sequence()
        all_samples = all_samples.simplify(filter_sites=False)
        geno = all_samples.genotype_matrix()
        non_missing = geno == -1
        mx = np.ma.masked_array(geno, mask=non_missing)
        cluster_centroids[cluster] = mx.sum(axis=1) / mx.count(axis=1)
    return torch.as_tensor(cluster_centroids)


def make_reproduction_tensor(
    clusters: torch.Tensor,
    *,
    crossover_rate: torch.Tensor,
    mutation_rate: torch.Tensor,
) -> torch.Tensor:
    """
    Computes pairwise conditional probabilities of sexual reproduction
    (crossover + mutation) using a pair HMM over genotypes.

    :param torch.Tensor clusters: A ``(num_variants, num_clusters)``-shaped
        array of clusters.
    :param torch.Tensor crossover_rate: A ``num_variants-1``-long vector of
        probabilties of crossover between successive variants.
    :param torch.Tensor mutation_rate: A ``num_variants``-long vector of
        mutation probabilites of mutation at each variant site.
    :returns: A reproduction tensor of shape ``(C, C, C)`` where ``C`` is the
        number of clusters. This tensor is symmetric in the first two axes and
        a normalized probability mass function over the third axis.
    :rtype: torch.Tensor
    """
    P, C = clusters.shape
    assert crossover_rate.shape == (P - 1,)
    assert mutation_rate.shape == (P,)

    # Construct a transition matrix.
    p = crossover_rate.neg().exp().mul(0.5).add(0.5)
    transition = torch.zeros(P - 1, 2, 2)
    transition[:, 0, 0] = p
    transition[:, 0, 1] = 1 - p
    transition[:, 1, 0] = 1 - p
    transition[:, 1, 1] = p

    # Construct an mutation matrix.
    p = mutation_rate.neg().exp().mul(0.5).add(0.5)
    mutate = torch.zeros(P, 2, 2)
    mutate[:, 0, 0] = p
    mutate[:, 0, 1] = 1 - p
    mutate[:, 1, 0] = 1 - p
    mutate[:, 1, 1] = p

    # Apply pair HMM along each genotype.
    result = torch.zeros(C, C, C)
    state = torch.full((C, C, C, 2), 0.5)
    for p in tqdm.tqdm(range(P)):
        # Update with observation + mutation noise.
        c = clusters[p].long()
        state[..., 0] *= mutate[p][c[:, None, None], c]
        state[..., 1] *= mutate[p][c[None, :, None], c]

        # Transition via crossover.
        if p < P - 1:
            state = state @ transition[p]

        # Numerically stabilize by moving norm to the result.
        total = state.sum(-1)
        state /= total.unsqueeze(-1)
        result += total.log()

    # Convert to a probability tensor.
    result -= result.logsumexp(-1, True)
    result.exp_()
    return result
