import math
from typing import Optional

import torch
from pyro.distributions import TorchDistribution
from torch.distributions import constraints


class ApproximateMatrixExponential:
    """
    A distance kernel based on a dense transition matrix.

    :param torch.Tensor transition: A stochastic matrix for
        right-multiplication, as in the update ``x = x @ transition``.
    :param int max_time_step: An integer upper bound on the number of
        transitions ever taken.
    """

    def __init__(
        self,
        transition: torch.Tensor,
        *,
        max_time_step: int = 60000,
        validate_args: bool = True,
    ):
        assert transition.dim() == 2
        assert transition.size(0) == transition.size(1)
        self.state_dim = transition.size(0)

        if validate_args:
            if not constraints.simplex.check(transition).all():
                raise ValueError(
                    "Expected transition to be a stochastic matrix, "
                    "but it was not normalized along the rightmost dimension."
                )

        # Create a collection of transition matrices of various powers of two.
        T = transition.detach()
        n = int(math.ceil(math.log2(max_time_step)))
        Ts = transition.new_zeros(n, *transition.shape)
        Ts[0] = T
        for i in range(1, n):
            Ts[i] = Ts[i - 1] @ Ts[i - 1]
        self.transitions = Ts

    def __call__(self, t: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        Stochastically transition a ``density`` vector by continuously-many
        time steps ``t``. For large time steps, the result is approximately::

            density @ torch.linalg.matrix_power(self.transition, t)

        but this supports fractional matrix powers, differentiably.
        """
        assert density.size(-1) == self.transitions.size(-1)
        t = torch.as_tensor(t, dtype=density.dtype)
        if t.dim():
            raise NotImplementedError("t cannot currently be batched")
        t0 = int(t.detach())
        t1 = 1 + t0

        # Compute transition up to t0.
        bits = list(map(int, reversed(bin(t0)[2:])))
        if len(bits) > len(self.transitions):
            raise ValueError(f"Time {t} exceeded max_time_step")  # TODO
        state = density
        for b, transition in zip(bits, self.transitions):
            if b:
                state = state @ transition

        # Interpolate from t0 up to t, supporting differentiation through t.
        state = (t1 - t) * state + (t - t0) * state @ self.transitions[0]

        return state


class WaypointDiffusion2D(TorchDistribution):
    """
    Example::

        class Model:
            # First compute a few static pieces of data.
            def __init__(self, ...):
                # These define the geography.
                self.waypoint_radius = ...
                self.waypoints = ...
                # This cheaply precomputes some matrices.
                self.matrix_exp = ApproximateMatrixExponential(...)

            # Then this
            def forward(self):
                ...
                # Concatenate observed and latent states.
                # Note both locations and times may be latent variables.
                locations = torch.cat([
                    leaf_locations,      # observed
                    internal_locations,  # latent
                ], dim-2)
                times = torch.cat([leaf_times, internal_times], dim=-1)

                # Add a likelihood term for migration.
                with pyro.plate("edges", num_edges):
                    pyro.sample(
                        "migration",
                        WaypointDiffusion2D(
                            source=location[edges.parent],
                            time=times[edges.child] - times[edges.parent],
                            radius=self.waypoint_radius,
                            waypoints=self.waypoints,
                            matrix_exp=self.matrix_exp,
                        ),
                        obs=location[edges.child],
                    )
    """

    arg_constraints = {"source": constraints.real_vector}
    support = constraints.real_vector

    def __init__(
        self,
        source: torch.Tensor,
        time: torch.Tensor,
        radius: torch.Tensor,
        waypoints: torch.Tensor,
        matrix_exp: ApproximateMatrixExponential,
        validate_args: Optional[bool] = None,
    ):
        assert source.dim() >= 1
        assert source.size(-1) == 2
        radius = torch.as_tensor(radius, dtype=source.dtype)
        assert radius.shape == ()
        assert waypoints.dim() == 2
        self.source = source
        self.time = time
        self.radius = radius
        self.waypoints = waypoints
        self.matrix_exp = matrix_exp
        super().__init__(
            batch_shape=source.shape[:-1],
            event_shape=source.shape[-1:],
            validate_args=validate_args,
        )

    def _waypoint_logp(self, position: torch.Tensor) -> torch.Tensor:
        assert position.size(-1) == 2
        r = torch.cdist(position, self.waypoints)
        return (
            -math.log(2 * math.pi)
            - self.radius.log()
            - 0.5 * (r / self.radius).square()
        )

    def log_prob(self, destin: torch.Tensor) -> torch.Tensor:
        finfo = torch.finfo(destin.dtype)
        source_logp = self._waypoint_logp(self.source)
        source_logp = source_logp - source_logp.logsumexp(-1, True)
        source_prob = source_logp.exp()
        destin_prob = self.matrix_exp(self.time, source_prob).clamp(min=finfo.tiny)
        destin_logp = destin_prob.log() + self._waypoint_logp(destin)
        return destin_logp.logsumexp(-1)
