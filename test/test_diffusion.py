import pytest
import scipy
import torch
from tspyro.diffusion import ApproximateMatrixExponential


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize("dim", [2, 3, 10])
def test_matrix_exp(dim, batch_shape):
    transition = torch.rand(dim, dim)
    transition /= transition.sum(-1, True)
    init = torch.rand(batch_shape + (dim,))
    init /= init.sum(-1, True)

    matrix_exp = ApproximateMatrixExponential(transition)

    for t in [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.5, 10.0, 100.0]:
        actual = matrix_exp(t, init)
        power = scipy.linalg.fractional_matrix_power(transition, t)
        power = torch.as_tensor(power, dtype=init.dtype)
        expected = init @ power
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=0.3 / (1 + t)), (actual, expected)
