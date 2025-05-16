from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from src.data import RKHSFnSampling
from src.rkhs import VectorKernel, LinearKernel, RKHSFn
from src.rkhs.testing import TestOutcome, ConditionedTestEmbedding, ConditionalTestEmbedding
from src.spec import KernelSpec, RKHSFnSpec, TestSpec, SpaceSpec, GaussianNoiseSpec, GaussianKernelSpec


@dataclass(frozen=True)
class Config:
    kernel: KernelSpec  # kernel on the input space
    regularization: float  # regularization parameter for the kernel ridge regression
    rkhs_fn: RKHSFnSpec  # configuration for the sampled RKHS functions
    dataset_size: int  # number of points to sample
    noise: GaussianNoiseSpec  # noise added to the function evaluations
    test: TestSpec[float]  # two-sample test parameters
    resolution: int  # number of points on which to evaluate the CMMD
    seed: int  # random seed

    SPACE = SpaceSpec(dim=1, min=-1, max=1)

    def sub_gaussian_std(self) -> float:
        return self.noise.std.max().item()

    def make_kernel(self) -> VectorKernel:
        return VectorKernel(x=self.kernel.make(), y=LinearKernel(ndim=0), regularization=self.regularization)

    def sample_rkhs_fn(self, kernel: VectorKernel, key: Array) -> RKHSFn:
        return self.rkhs_fn.sample(kernel.x, self.SPACE, key=key, surface=True, shape=())

    def sample_dataset(self, fn: RKHSFn, key: Array) -> RKHSFnSampling:
        key_xs, key_noise = jax.random.split(key)

        xs = self.SPACE.uniform_sample(self.dataset_size, key_xs)
        ys = fn(xs)

        noise = self.noise.sample(ys.shape, key_noise)

        return RKHSFnSampling(xs=xs, ys=ys + noise)


@dataclass(frozen=True)
class Result:
    results: TestOutcome
    fn_1: RKHSFn
    fn_2: RKHSFn
    cme_1: ConditionalTestEmbedding
    cme_2: ConditionalTestEmbedding
    kmes_1: ConditionedTestEmbedding
    kmes_2: ConditionedTestEmbedding

    def rejection_intervals(self, states: Array) -> list[tuple[float, float, bool]]:
        reject = self.results.rejection()
        intervals = []

        x = float(states[0].item())

        for i in range(1, len(states)):
            if reject[i] != reject[i - 1]:
                state = float(states[i].item())
                intervals.append((x, state, not bool(reject[i])))
                x = state

        if not intervals:
            return [(
                float(states[0].item()),
                float(states[-1].item()),
                bool(reject[-1])
            )]

        last_state = float(states[-1].item())

        if intervals[-1][1] != last_state:
            intervals.append((x, last_state, bool(reject[-1])))

        return intervals


DEFAULT_NOISE_STD = 0.05

DEFAULT_ARGS = dict(
    kernel=GaussianKernelSpec(0.25, ndim=1),
    regularization=0.01,
    rkhs_fn=RKHSFnSpec(n_basis_points=12, ball_radius=1.0),
    dataset_size=25,
    noise=GaussianNoiseSpec(mean=jnp.array([0.0]), std=jnp.array([DEFAULT_NOISE_STD])),
    resolution=2000,
    seed=3
)
