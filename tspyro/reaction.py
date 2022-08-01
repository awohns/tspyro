from typing import Callable
from typing import Optional

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.reparam import DiscreteCosineReparam
from pyro.infer.reparam import HaarReparam
from pyro.optim import ClippedAdam


class SingleChromosomeReaction:
    """
    :param torch.Tensor reproduction_op: A `(C,C,C)` shaped array from
        :func:`tspyro.cluster.make_reproduction_tensor`
    :param torch.Tensor leaf_times: A `N`-length array of sample times in units of
        generations.
    :param torch.Tensor leaf_clusters: A `N`-length array of cluster ids, each
        of which takes a value in `[0,C-1]`.
    """

    latent_variables = ["diffusion"]

    def __init__(
        self,
        reproduction_op: torch.Tensor,
        leaf_times: torch.Tensor,
        leaf_clusters: torch.Tensor,
        num_time_steps: int,
    ):
        super().__init__()

        # These are defined in .fit().
        self._model: Optional[Callable] = None
        self._guide: Optional[AutoNormal] = None

        # Validate data.
        assert reproduction_op.dtype == torch.float
        assert leaf_times.dtype == torch.long
        assert leaf_clusters.dtype == torch.long
        C = len(reproduction_op)
        N = len(leaf_times)
        T = num_time_steps
        assert reproduction_op.shape == (C, C, C)
        assert leaf_times.shape == (N,)
        assert leaf_clusters.shape == (N,)
        assert 0 <= leaf_times.max().item() < T
        assert 0 <= leaf_clusters.max().item() < C
        self.reproduction_op = reproduction_op
        self.leaf_times = leaf_times
        self.leaf_clusters = leaf_clusters
        self.num_time_steps = num_time_steps

    @staticmethod
    def model(
        reproduction_op: torch.Tensor,
        leaf_times: torch.Tensor,
        leaf_clusters: torch.Tensor,
        num_time_steps: int,
    ):
        T = num_time_steps
        C = len(reproduction_op)
        N = len(leaf_times)
        time_plate = pyro.plate("time", T, dim=-2)
        step_plate = pyro.plate("step", T - 1, dim=-2)
        cluster_plate = pyro.plate("cluster", C, dim=-1)
        leaf_plate = pyro.plate("leaves", N, dim=-1)

        # Sample density from an improper prior.
        # Consider instead using an informative prior.
        # Consider using HaarReparam(dim=-2) or DiscreteCosineReparam(dim=-2).
        with time_plate, cluster_plate, poutine.mask(mask=False):
            density = pyro.sample("density", dist.Exponential(1))
            density = density.clamp(min=1e-6)

        # Reaction factor.
        # Note technically there are dependencies across cluster_plate.
        with step_plate, cluster_plate:
            parent_dist = density[:-1]
            child_dist = density[1:]
            prediction = torch.einsum(
                "tc,td,cde->te",
                parent_dist,  # [T,C]
                parent_dist,  # [T,C]
                reproduction_op,  # [C,C,C] Applies crossover.
            )  # [T,C]
            # Renormalize after selection_op.
            prediction = prediction / prediction.sum(-1, True)
            pyro.sample(
                "reaction",
                dist.Normal(prediction, prediction.sqrt()),  # approximate multinomial
                obs=child_dist,
            )

        # Observation of samples with known (time, genome).
        with leaf_plate:
            pyro.factor("obs", density[leaf_times, leaf_clusters].log())

    @staticmethod
    def config_reparam(self, rep: Optional[str] = None):
        config = {}
        if rep == "haar":
            config["diffusion"] = HaarReparam(dim=-2)
        elif rep == "dct":
            config["diffusion"] = DiscreteCosineReparam(dim=-2)
        elif rep:
            raise ValueError(f"Unknownn reparam: {rep}")
        return config

    def fit(
        self,
        *,
        reparam: Optional[dict] = None,
        num_steps: int = 10001,
        learning_rate: float = 1e-2,
        learning_rate_decay: float = 1e-1,
        log_every: int = 100,
    ) -> dict:
        self._model = self.model

        # Optionally reparametrize model.
        if reparam:
            self._model = poutine.reparam(self._model, reparam)

        # Fit using SVI.
        self._guide = AutoNormal(self._model, init_scale=0.01)
        pyro.clear_param_store()
        optim = ClippedAdam(
            {"lr": learning_rate, "lrd": learning_rate_decay ** (1 / num_steps)}
        )
        elbo = Trace_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        self.losses = []
        args = (
            self.reproduction_op,
            self.leaf_times,
            self.leaf_clusters,
            self.num_time_steps,
        )
        for step in range(num_steps):
            loss = svi.step(*args)
            self.losses.append(loss)
            if log_every and step % log_every == 0:
                print(f"step {step} loss = {loss:0.6g}")

        # Predict.
        with torch.no_grad(), poutine.block():
            median = self._guide.median()
            trace = poutine.trace(poutine.condition(self._model, median)).get_trace(
                *args
            )
        latents = ["diffusion"]
        self.prediction = {trace.nodes[name]["value"] for name in latents}

        return {"losses": self.losses, "prediction": self.prediction}


def single_chromosome_reaction_model(
    reproduction_op: torch.Tensor,
    leaf_times: torch.Tensor,
    leaf_clusters: torch.Tensor,
    num_time_steps: int,
):
    assert reproduction_op.dtype == torch.float
    assert leaf_times.dtype == torch.long
    assert leaf_clusters.dtype == torch.long
    T = num_time_steps
    C = len(reproduction_op)
    N = len(leaf_times)
    assert reproduction_op.shape == (C, C, C)
    assert leaf_times.shape == (N,)
    assert leaf_clusters.shape == (N,)
    assert leaf_times.max().item() < num_time_steps
    time_plate = pyro.plate("time", T, dim=-2)
    step_plate = pyro.plate("step", T - 1, dim=-2)
    cluster_plate = pyro.plate("cluster", C, dim=-1)
    leaf_plate = pyro.plate("leaves", N, dim=-1)

    # Sample density from an improper prior.
    # Consider instead using an informative prior.
    # Consider using HaarReparam(dim=-3) or DiscreteCosineReparam(dim=-3).
    with time_plate, cluster_plate, pyro.mask(mask=False):
        density = pyro.sample("density", dist.Exponential(1))

    # Reaction factor.
    # Note technically there are dependencies across cluster_plate.
    with step_plate, cluster_plate:
        parent_dist = density[:-1]
        child_dist = density[1:]
        prediction = torch.einsum(
            "tc,td,cde->te",
            parent_dist,  # [T,C]
            parent_dist,  # [T,C]
            reproduction_op,  # [C,C,C] Applies crossover.
        )  # [T,C]
        # Renormalize after selection_op.
        prediction = prediction / prediction.sum(-1, True)
        pyro.sample(
            "reaction",
            dist.Normal(prediction, prediction.sqrt()),  # approximate multinomial
            obs=child_dist,
        )

    # Observation of samples with known (time, genome).
    with leaf_plate:
        pyro.factor("obs", density[leaf_times, leaf_clusters].log())
