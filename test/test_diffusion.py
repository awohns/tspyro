import pytest
import scipy
import torch
from tspyro.diffusion import ApproximateMatrixExponential
from tspyro.diffusion import WaypointDiffusion2D


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
        assert torch.allclose(actual, expected, atol=0.5 / (1 + t)), (actual, expected)


@pytest.mark.parametrize("batch_time", [False, True])
@pytest.mark.parametrize("batch_radius", [False, True])
def test_waypoint_diffusion_smoke(batch_time, batch_radius):
    batch_shape = (11,)
    X, Y = 4, 5
    XY = X * Y
    waypoints = torch.stack(
        [
            torch.arange(float(X))[:, None].expand((X, Y)).reshape(-1),
            torch.arange(float(Y)).expand((X, Y)).reshape(-1),
        ],
        dim=-1,
    )
    source = torch.randn(batch_shape + (2,)) + 2
    destin = torch.randn(batch_shape + (2,)) + 2
    time = 10.0 * torch.randn(batch_shape if batch_time else ()).exp()
    radius = torch.rand(len(waypoints)) if batch_radius else 1.0
    transition = torch.rand(XY, XY)
    transition /= transition.sum(-1, True)
    matrix_exp = ApproximateMatrixExponential(transition)

    d = WaypointDiffusion2D(source, time, radius, waypoints, matrix_exp)
    actual = d.log_prob(destin)
    assert torch.isfinite(actual).all()
    assert actual.shape == batch_shape
