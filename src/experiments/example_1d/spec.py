from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.config import DataConfig, IIDDataConfig, SpaceConfig, GaussianNoiseConfig, SubGaussianNoiseConfig
from src.data import RKHSFnSampling
from src.random import uniform_rkhs_subspace_fn
from src.rkhs import RKHSFn, Kernel, LinearKernel, GaussianKernel
from src.util import KernelParametrization


@dataclass
class Config:
    kernel_x_bandwidth: float  # bandwidth of the Gaussian kernel for the input space
    regularization: float  # regularization parameter for the kernel ridge regression
    n_basis_points: int  # number of points on which each function is centered
    rkhs_ball_radius: float  # radius of the RKHS ball from which the functions are sampled
    data: DataConfig  # configuration for generating the datasets
    noise: SubGaussianNoiseConfig  # configuration for adding noise to the datasets
    n_bootstrap: int  # number of bootstrap datasets to generate
    confidence_level: float  # confidence level for the hypothesis test
    resolution: int  # number of points on which to evaluate the CMMD
    seed: int  # random seed

    def __post_init__(self):
        assert self.kernel_x_bandwidth > 0
        assert self.data.space.dim == 1
        assert self.n_bootstrap >= 1
        assert 0 < self.confidence_level < 1

    def kernel_parametrization(self) -> KernelParametrization:
        return KernelParametrization(
            x=GaussianKernel(bandwidth=self.kernel_x_bandwidth),
            y=LinearKernel(ndim=0),
            regularization=self.regularization
        )

    def state_space(self) -> jnp.ndarray:
        return self.data.space.discretization(self.resolution)

    def sample_rkhs_fn(self, kernel: Kernel, key: jax.Array) -> RKHSFn:
        key_points, key_fn = jax.random.split(key)

        return uniform_rkhs_subspace_fn(
            kernel=kernel,
            xs=self.data.space.uniform_sample(self.n_basis_points, key_points),
            radius=self.rkhs_ball_radius,
            key=key_fn
        )

    def sample_dataset(self, kernel: Kernel, fn: RKHSFn, key: jax.Array) -> RKHSFnSampling:
        key_xs, key_noise = jax.random.split(key)
        xs = self.data.sample(key)
        ys = kernel.evaluate.one_many(fn, xs)
        noise = self.noise.sample(ys.shape, key_noise)
        return RKHSFnSampling(xs=xs, ys=ys + noise)


@dataclass
class Result:
    dataset_1: RKHSFnSampling
    dataset_2: RKHSFnSampling
    cmmds: jnp.ndarray
    beta_1_botstrap: jnp.ndarray
    beta_2_bootstrap: jnp.ndarray
    beta_1_analytical: jnp.ndarray
    beta_2_analytical: jnp.ndarray
    sigmas_1: jnp.ndarray
    sigmas_2: jnp.ndarray
    values_1: jnp.ndarray
    values_2: jnp.ndarray
    estimated_values_1: jnp.ndarray
    estimated_values_2: jnp.ndarray

    def thresholds_bootstrap(self) -> jnp.ndarray:
        return self.beta_1_botstrap * self.sigmas_1 + self.beta_2_bootstrap * self.sigmas_2

    def thresholds_analytical(self) -> jnp.ndarray:
        return self.beta_1_analytical * self.sigmas_1 + self.beta_2_analytical * self.sigmas_2

    def rejection_intervals(self, states: jnp.ndarray, thresholds: jnp.ndarray) -> list[tuple[float, float, bool]]:
        reject = self.cmmds > thresholds
        intervals = []

        x = float(states[0].item())

        for i in range(1, len(states)):
            if reject[i] != reject[i - 1]:
                state = float(states[i].item())
                intervals.append((x, state, not bool(reject[i])))
                x = state

        last_state = float(states[-1].item())

        if intervals[-1][1] != last_state:
            intervals.append((x, last_state, bool(reject[-1])))

        return intervals


DEFAULT_CONFIG = Config(
    kernel_x_bandwidth=0.2,
    regularization=0.01,
    n_basis_points=12,
    rkhs_ball_radius=1.0,
    data=IIDDataConfig(
        space=SpaceConfig(dim=1, min=-1, max=1),
        dataset_size=40
    ),
    noise=GaussianNoiseConfig(mean=jnp.array([0.0]), std=jnp.array([0.05])),
    resolution=2000,
    n_bootstrap=1000,
    confidence_level=0.05,
    seed=3
)
