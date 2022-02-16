import argparse

import pytest
import torch
from tspyro.cluster import check_sparse_genotypes
from tspyro.cluster import make_clustering_gibbs


def make_fake_data(num_samples, num_variants):
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


@pytest.mark.parametrize("num_epochs", [1, 10])
@pytest.mark.parametrize("num_clusters", [5])
@pytest.mark.parametrize("num_samples", [5, 10, 100])
@pytest.mark.parametrize("num_variants", [20])
def test_clustering_gibbs(num_variants, num_samples, num_clusters, num_epochs):
    print(f"Making fake data of size {num_samples} x {num_variants}")
    data = make_fake_data(num_samples, num_variants)
    print(f"Creating {num_clusters} clusters via {num_epochs} epochs")
    clusters = make_clustering_gibbs(data, num_clusters, num_epochs=num_epochs)
    assert clusters.shape == (num_variants, num_clusters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark clustering algorithm")
    parser.add_argument("-n", "--num-samples", default=1000, type=int)
    parser.add_argument("-p", "--num-variants", default=1000, type=int)
    parser.add_argument("-k", "--num-clusters", default=100, type=int)
    parser.add_argument("-e", "--num-epochs", default=10, type=int)
    args = parser.parse_args()

    test_clustering_gibbs(
        args.num_variants, args.num_samples, args.num_clusters, args.num_epochs
    )
