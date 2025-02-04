from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.config import BootstrapThresholdConfig, RotationDataConfig, SpaceConfig, IIDDataConfig, \
    KernelParametrizationConfig, GaussianKernelConfig, GaussianNoiseConfig, DataConfig, GaussianMixtureNoiseConfig, \
    LinearKernelConfig
from src.data import RKHSFnSampling, collect_rotation_trajectories
from src.random import uniform_rkhs_subspace_fn
from src.rkhs import RKHSFn, Kernel


@dataclass(frozen=True)
class Config:
    kernel: KernelParametrizationConfig  # kernel configuration
    threshold: BootstrapThresholdConfig  # configuration for bootstrapping of the CMMD thresholds
    dataset_size: int  # number of samples in each dataset
    noise: GaussianNoiseConfig  # noise added to the function values
    resolution: int  # number of points per dimension for the discretization of the space
    seed: int  # random seed

    @property
    def space(self):
        return SpaceConfig(dim=2, min=-4, max=4)

    def state_space(self) -> jnp.ndarray:
        return self.space.discretization(self.resolution)

    def sample_rkhs_fns(self, kernel: Kernel, key: jax.Array) -> tuple[RKHSFn, RKHSFn]:
        key_11, key_12, key_21, key_22 = jax.random.split(key, 4)

        fn_1 = uniform_rkhs_subspace_fn(
            kernel=kernel,
            xs=self.space.uniform_sample(11, key_11),
            radius=1,
            key=key_12
        )

        fn_2 = uniform_rkhs_subspace_fn(
            kernel=kernel,
            xs=self.space.uniform_sample(11, key_21),
            radius=1,
            key=key_22
        )

        return fn_1, fn_2

    def sample_noisy_dataset(self, kernel: Kernel, fn: RKHSFn, xs: jnp.ndarray, key: jax.Array) -> RKHSFnSampling:
        assert xs.ndim == 2
        assert xs.shape[1] == 2

        ys = kernel.evaluate.one_many(fn, xs)
        noise = self.noise.sample(ys.shape, key)

        return RKHSFnSampling(xs=xs, ys=ys + noise)

    def sample_noisy_iid_dataset(self, kernel: Kernel, fn: RKHSFn, key: jax.Array) -> RKHSFnSampling:
        xs = self.space.uniform_sample(self.dataset_size, key)
        return self.sample_noisy_dataset(kernel, fn, xs, key)

    def sample_noisy_rotation_dataset(self, kernel: Kernel, fn: RKHSFn, key: jax.Array) -> RKHSFnSampling:
        key_init, key_noise = jax.random.split(key)

        # ensure that we only have data far from the center
        # to that end, sample frm MoG located at the corners of the space
        init_distribution = GaussianMixtureNoiseConfig(
            mean=jnp.array([[2.5, 2.5], [2.5, -2.5], [-2.5, -2.5], [-2.5, 2.5]]),
            std=jnp.array([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]]),
        )

        xs = collect_rotation_trajectories(
            xs_init=init_distribution.sample(4, key_init),
            n_rotations=0.5,
            n_samples=self.dataset_size // 4,
            damping_factor=0.5
        ).reshape(-1, self.space.dim)

        return self.sample_noisy_dataset(kernel, fn, xs, key_noise)


@dataclass(frozen=True)
class Result:
    cmmd_iid: jnp.ndarray
    cmmd_rotation: jnp.ndarray
    thresholds_iid: jnp.ndarray
    thresholds_rotation: jnp.ndarray

    def rejection_region_iid(self) -> jnp.ndarray:
        return self.cmmd_iid > self.thresholds_iid

    def rejection_region_rotation(self) -> jnp.ndarray:
        return self.cmmd_rotation > self.thresholds_rotation


DEFAULT_CONFIG = Config(
    kernel=KernelParametrizationConfig(
        x_config=GaussianKernelConfig(bandwidth=1, ndim=1),
        y_config=GaussianKernelConfig(bandwidth=0.5, ndim=0),
        regularization=0.01
    ),
    threshold=BootstrapThresholdConfig(
        confidence_level=0.05,
        single_beta=False,
        n_bootstrap=100
    ),
    dataset_size=100,
    noise=GaussianNoiseConfig(mean=jnp.array([0]), std=jnp.array([0.01])),
    resolution=200,
    seed=0
)
