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

    return offsets, index, values


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
    K = num_clusters

    counts = torch.full((P, K, 2), 0.5, dtype=torch.long)
    assignment = torch.full((N,), -1, dtype=torch.long)

    # Use a shuffled linearly annealed schedule.
    schedule = (([+1, -1] * num_epochs)[:-1]) * N
    shuffle = torch.randperm(N)
    assert sum(schedule) == N
    pending = {+1: 0, -1: 0}
    for sign in tqdm.tqdm(schedule):
        # Select the next pending datum.
        n = shuffle[pending[sign] % N]
        pending[sign] += 1
        beg, end = offsets[n]
        index_n = index[beg:end]
        value_n = values[beg:end]

        if sign > 0:
            # Add the datum to a random cluster.
            assert assignment[n] == -1
            posterior = counts[index_n].float().add_(0.5)  # add Jeffreys prior
            posterior /= posterior.sum(-1, True)
            tails, heads = posterior.unbind(-1)
            logits = torch.where(value_n[:, None], heads, tails).sum(0)
            logits -= logits.max()
            k = int(logits.exp().multinomial(1))
            assignment[n] = k
        else:
            # Remove the datum from the current cluster.
            assert assignment[n] != -1
            k = int(assignment[n])
            assignment[n] = -1

        counts[index_n, k, value_n.long()] += sign
    assert all(assignment >= 0)

    posterior = counts.float().add_(0.5)  # add Jeffreys prior
    clusters = (posterior[..., 1] / posterior.sum(-1)).round().bool()
    return clusters
