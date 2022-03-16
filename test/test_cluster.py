import argparse

import pytest
import torch
from tspyro.cluster import make_clustering_gibbs
from tspyro.cluster import make_fake_data
from tspyro.cluster import make_reproduction_tensor
from tspyro.cluster import transpose_sparse


@pytest.mark.parametrize("num_epochs", [1, 10])
@pytest.mark.parametrize("num_clusters", [5])
@pytest.mark.parametrize("num_samples", [5, 10, 100])
@pytest.mark.parametrize("num_variants", [20])
def test_clustering(num_variants, num_samples, num_clusters, num_epochs):
    print(f"Making fake data of size {num_samples} x {num_variants}")
    data = make_fake_data(num_samples, num_variants)

    print(f"Creating {num_clusters} clusters via {num_epochs} epochs")
    assignment, clusters = make_clustering_gibbs(
        data, num_clusters, num_epochs=num_epochs
    )
    assert assignment.shape == (num_samples,)
    assert clusters.shape == (num_variants, num_clusters)

    print("Creating a reproduction tensor")
    crossover_rate = torch.rand(num_variants - 1) * 0.01
    mutation_rate = 0.001
    reproduce = make_reproduction_tensor(
        clusters,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
    )
    assert isinstance(reproduce, torch.Tensor)
    assert reproduce.shape == (num_clusters, num_clusters, num_clusters)
    assert torch.allclose(reproduce, reproduce.transpose(0, 1))
    assert torch.allclose(reproduce.sum(-1), torch.ones(num_clusters, num_clusters))


@pytest.mark.parametrize("num_samples", [5, 10, 100])
@pytest.mark.parametrize("num_variants", [4, 8, 100])
def test_transpose(num_variants, num_samples):
    data1 = make_fake_data(num_samples, num_variants)
    data2 = transpose_sparse(data1)
    data3 = transpose_sparse(data2)
    for k, v1 in data1.items():
        v3 = data3[k]
        assert (v1 == v3).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark clustering algorithm")
    parser.add_argument("-n", "--num-samples", default=1000, type=int)
    parser.add_argument("-p", "--num-variants", default=1000, type=int)
    parser.add_argument("-k", "--num-clusters", default=100, type=int)
    parser.add_argument("-e", "--num-epochs", default=10, type=int)
    args = parser.parse_args()

    test_clustering(
        args.num_variants, args.num_samples, args.num_clusters, args.num_epochs
    )
