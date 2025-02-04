from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.config import KernelParametrizationConfig, SpaceConfig, GaussianKernelConfig, RotationDataConfig, \
    GaussianNoiseConfig
from src.data import RKHSFnSampling, collect_rotation_trajectories
from src.random import uniform_rkhs_subspace_fn
from src.rkhs import Kernel, RKHSFn

DAMPING_FACTOR = 0.1

N_ROTATIONS = 0.5

N_BASIS_ITEMS = 12  # number of partial kernel evaluations used to generate RKHS functions


@dataclass
class Config:
    kernel: KernelParametrizationConfig  # kernel parameters
    size_reference: int  # number of reference points
    size_window: int  # size of the window
    n_evaluations: int  # number of most recent points to evaluate at
    disturbance: float  # RKHS norm of the disturbance function added to the true function
    noise: GaussianNoiseConfig  # Gaussian noise parameters
    n_bootstrap: int  # number of bootstrap datasets for the CMMD threshold
    confidence_level: float  # confidence level for the CMMD threshold
    seed: int  # random seed

    @property
    def space(self) -> SpaceConfig:
        return SpaceConfig(dim=2, min=-4, max=4)

    def sample_true_fn(self, kernel: Kernel, key: jax.Array) -> RKHSFn:
        return uniform_rkhs_subspace_fn(
            kernel=kernel,
            xs=self.space.uniform_sample(N_BASIS_ITEMS, key),
            radius=1,
            key=key,
        )

    def sample_disturbance_fn(self, kernel: Kernel, key: jax.Array) -> RKHSFn:
        return uniform_rkhs_subspace_fn(
            kernel=kernel,
            xs=self.space.uniform_sample(N_BASIS_ITEMS, key),
            radius=self.disturbance,
            key=key,
            sphere=True
        )

    def sample_noisy_fn_evaluations(
            self,
            kernel: Kernel,
            fn: RKHSFn,
            xs: jnp.ndarray,
            key: jax.Array
    ) -> RKHSFnSampling:
        ys = kernel.evaluate.one_many(fn, xs)
        noise = self.noise.sample(ys.shape, key)
        return RKHSFnSampling(xs=xs, ys=ys + noise)

    def sample_reference_dataset(self, kernel: Kernel, fn: RKHSFn, key: jax.Array) -> RKHSFnSampling:
        key_xs, key_noise = jax.random.split(key)

        xs = RotationDataConfig(
            space=self.space,
            n_initializations=10,
            trajectory_length=self.size_reference // 10,
            n_rotations=N_ROTATIONS,
            damping_factor=DAMPING_FACTOR
        ).sample(key_xs)
        return self.sample_noisy_fn_evaluations(kernel, fn, xs, key_noise)

    def sample_trajectory(
            self,
            kernel: Kernel,
            fn: RKHSFn,
            x_init: jnp.ndarray,
            length: int,
            key: jax.Array
    ) -> RKHSFnSampling:
        xs = collect_rotation_trajectories(
            xs_init=x_init.reshape(1, -1),
            n_rotations=N_ROTATIONS / (self.size_reference // 10) * length,
            n_samples=length,
            damping_factor=DAMPING_FACTOR
        ).reshape(-1, 2)

        return self.sample_noisy_fn_evaluations(kernel, fn, xs, key)


@dataclass(frozen=True)
class Result:
    cmmd_windows: jnp.ndarray
    threshold_windows: jnp.ndarray


DEFAULT_KERNEL_CONFIG = KernelParametrizationConfig(
    x_config=GaussianKernelConfig(bandwidth=1, ndim=1),
    y_config=GaussianKernelConfig(bandwidth=0.5, ndim=0),
    regularization=0.01
)

DEFAULT_ARGS = dict(
    kernel=DEFAULT_KERNEL_CONFIG,
    size_reference=100,
    size_window=20,
    n_evaluations=20,
    noise=GaussianNoiseConfig(mean=jnp.array([0]), std=jnp.array([0.01])),
    n_bootstrap=100,
    confidence_level=0.05,
    seed=0
)
