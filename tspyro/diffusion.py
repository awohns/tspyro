import math

import torch
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
