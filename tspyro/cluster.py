from typing import Tuple

import numpy as np
import torch
import tqdm


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

    assert offsets[0, 0] == 0
    assert (offsets[1:, 0] == offsets[:-1, 1]).all()

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


def transpose_sparse(data: dict, num_cols=0) -> dict:
    """
    Convert from row-oriented to column-oriented.
    This function is its own inverse.

    :param dict data: A dict representing sparse genotypes.
    :param int num_cols: Optional number of columns.
    :returns: A dict representing sparse genotypes.
    :rtype: dict
    """
    old_offsets, old_index, old_values = check_sparse_genotypes(data)
    num_rows = len(old_offsets)
    num_cols = max(num_cols, 1 + int(old_index.max()))

    # Initialize positions and create offsets.
    position = torch.zeros(1 + num_cols, dtype=torch.long)
    for r in range(num_rows):
        beg, end = old_offsets[r].tolist()
        index = old_index[beg:end]
        position[1 + index] += 1
    position.cumsum_(0)
    new_offsets = torch.stack([position[:-1], position[1:]], dim=-1)

    # Populate index and values.
    new_index = torch.zeros_like(old_index)
    new_values = torch.zeros_like(old_values)
    for r in range(num_rows):
        beg, end = old_offsets[r].tolist()
        index = old_index[beg:end]
        values = old_values[beg:end]
        new_index[position[index]] = r
        new_values[position[index]] = values
        position[index] += 1

    return dict(offsets=new_offsets, index=new_index, values=new_values)


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
    for haplo in tqdm(ts.haplotypes(), total=ts.num_samples):
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
    for var in tqdm(ts.variants(), total=ts.num_sites):
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
) -> Tuple[torch.Tensor, torch.Tensor]:
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
