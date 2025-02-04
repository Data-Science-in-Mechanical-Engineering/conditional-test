import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp

from src.config import DataConfig, IIDDataConfig, RotationDataConfig, AnalyticalThresholdConfig
from src.config import KernelParametrizationConfig, KernelConfig, LinearKernelConfig, GaussianKernelConfig
from src.config import NoiseConfig, GaussianMixtureNoiseConfig
from src.config import SpaceConfig
from src.config import ThresholdConfig, BootstrapThresholdConfig
from src.data import RKHSFnSampling
from src.random import uniform_rkhs_subspace_fns
from src.rkhs import RKHSFn, Kernel, GaussianKernel


@dataclass(frozen=True)
class Config(ABC):
    kernel_x_config: KernelConfig  # kernel on the input space
    regularization: float  # regularization parameter for the kernel
    n_functions: int  # number of functions to sample from the RKHS
    n_basis_points: int  # number of points on which each function is centered
    rkhs_ball_radius: float  # radius of the RKHS ball from which the functions are sampled
    data: DataConfig  # configuration for generating the datasets
    noise_1_top: GaussianMixtureNoiseConfig  # noise on dataset 1 in the top half of the state space
    noise_1_bottom: GaussianMixtureNoiseConfig  # noise on dataset 1 in the bottom half of the state space
    noise_2_top: GaussianMixtureNoiseConfig  # noise on dataset 2 in the top half of the state space
    noise_2_bottom: GaussianMixtureNoiseConfig  # noise on dataset 2 in the bottom half of the state space
    resolution: int  # number of points in each dimension of the grid
    seed: int  # random seed

    @abstractmethod
    def parametrization(self) -> KernelParametrizationConfig:
        raise NotImplementedError

    def __post_init__(self):
        assert self.regularization > 0
        assert self.n_functions >= 1
        assert self.n_basis_points >= 1
        assert self.rkhs_ball_radius > 0
        assert self.data.space.dim == 2
        assert self.resolution >= 1

    @staticmethod
    def state_space_noise_mask(x: jnp.ndarray) -> jnp.ndarray:
        return x[..., 0] < 0

    def sample_rkhs_fns(self, kernel: Kernel, key: jax.Array) -> RKHSFn:
        key_xs, key_fn = jax.random.split(key)

        return uniform_rkhs_subspace_fns(
            kernel=kernel,
            xs_batch=self.data.space.uniform_sample((self.n_functions, self.n_basis_points), key_xs),
            radius=self.rkhs_ball_radius,
            key=key_fn
        )

    def sample_function_evaluations(self, kernel_fn: Kernel, functions: RKHSFn, key: jax.Array) -> RKHSFnSampling:
        key_data, key_noise = jax.random.split(key)

        n_functions = functions.coefficients.shape[0]

        @partial(jax.vmap)
        def batch_fn(fn: RKHSFn, key_: jax.Array) -> RKHSFnSampling:
            return RKHSFnSampling.at_points(
                fn=fn, kernel=kernel_fn,
                xs=self.data.sample(key_)
            )

        keys = jax.random.split(key_data, n_functions)
        return batch_fn(functions, keys)

    def sample_noisy_datasets(
            self,
            kernel_fn: Kernel, fns: RKHSFn,
            noise_top: NoiseConfig, noise_bottom: NoiseConfig,
            key: jax.Array
    ) -> RKHSFnSampling:
        key_1, key_2, key_3 = jax.random.split(key, 3)

        dataset = self.sample_function_evaluations(kernel_fn, fns, key_1)

        mask = self.state_space_noise_mask(dataset.xs)

        noise_top = noise_top.sample(dataset.ys.shape, key_2)
        noise_bottom = noise_bottom.sample(dataset.ys.shape, key_3)
        noise = noise_top * mask + noise_bottom * ~mask

        noisy_ys = dataset.ys + noise

        return RKHSFnSampling(xs=dataset.xs, ys=noisy_ys)


@dataclass(frozen=True)
class MeanConfig(Config):
    threshold: ThresholdConfig  # configuration for the computation of the decision threshold on the CMMD

    def parametrization(self) -> KernelParametrizationConfig:
        return KernelParametrizationConfig(
            x_config=self.kernel_x_config,
            y_config=LinearKernelConfig(ndim=0),
            regularization=self.regularization
        )


@dataclass(frozen=True)
class DistributionalConfig(Config):
    kernel_y_bandwidth: float  # bandwidth of the Gaussian kernel on the output space
    threshold: BootstrapThresholdConfig  # configuration for the computation of the decision threshold on the CMMD

    def __post_init__(self):
        super().__post_init__()
        assert type(self.noise_1_top) == type(self.noise_2_top)

    def parametrization(self) -> KernelParametrizationConfig[Kernel, GaussianKernel]:
        return KernelParametrizationConfig(
            x_config=self.kernel_x_config,
            y_config=GaussianKernelConfig(bandwidth=self.kernel_y_bandwidth, ndim=0),
            regularization=self.regularization
        )


@dataclass(frozen=True)
class Result:
    functions: RKHSFn
    datasets_1: RKHSFnSampling
    datasets_2: RKHSFnSampling
    cmmd_grid_pairwise: jnp.ndarray
    fn_values_grid_pairwise: jnp.ndarray
    true_cmmd_grid_pairwise: jnp.ndarray
    threshold_grid_pairwise: jnp.ndarray
    rejection_grid_pairwise: jnp.ndarray

    @property
    def cmmd_max_scale(self) -> float:
        return max(
            float(self.rejection_grid_pairwise.max()),
            float(self.cmmd_grid_pairwise.max()),
            float(self.threshold_grid_pairwise.max())
        )


CONFIG_MEAN_IID = MeanConfig(
    kernel_x_config=GaussianKernelConfig(bandwidth=1, ndim=1),
    regularization=0.1,
    n_functions=4,
    n_basis_points=12,
    rkhs_ball_radius=1.0,
    data=IIDDataConfig(
        space=SpaceConfig(dim=2, min=-4, max=4),
        dataset_size=250
    ),
    noise_1_top=GaussianMixtureNoiseConfig(
        mean=jnp.array([0.0]),
        std=jnp.array([0.01])
    ),
    noise_1_bottom=GaussianMixtureNoiseConfig(
        mean=jnp.array([0.0]),
        std=jnp.array([0.01])
    ),
    noise_2_top=GaussianMixtureNoiseConfig(
        mean=jnp.array([0.0]),
        std=jnp.array([0.01])
    ),
    noise_2_bottom=GaussianMixtureNoiseConfig(
        mean=jnp.array([0.0]),
        std=jnp.array([0.01])
    ),
    resolution=50,
    threshold=AnalyticalThresholdConfig(confidence_level=0.05),
    seed=0
)

CONFIG_MEAN_TRAJECTORY = dataclasses.replace(
    CONFIG_MEAN_IID,
    data=RotationDataConfig(
        space=SpaceConfig(dim=2, min=-4, max=4),
        n_initializations=2,
        trajectory_length=50,
        n_rotations=2,
        damping_factor=0.4
    )
)

CONFIG_DISTRIBUTIONAL_IID = DistributionalConfig(
    kernel_x_config=GaussianKernelConfig(bandwidth=1, ndim=1),
    kernel_y_bandwidth=0.25,
    regularization=0.1,
    n_functions=4,
    n_basis_points=12,
    rkhs_ball_radius=1,
    data=IIDDataConfig(
        space=SpaceConfig(dim=2, min=-4, max=4),
        dataset_size=250
    ),
    noise_1_top=GaussianMixtureNoiseConfig(
        mean=jnp.array([0.0]),
        std=jnp.array([0.01])
    ),
    noise_1_bottom=GaussianMixtureNoiseConfig(
        mean=jnp.array([0.0]),
        std=jnp.array([0.01])
    ),
    noise_2_top=GaussianMixtureNoiseConfig(
        mean=jnp.array([-0.5, 0.5]),
        std=jnp.array([0.01, 0.01])
    ),
    noise_2_bottom=GaussianMixtureNoiseConfig(
        mean=jnp.array([0.0]),
        std=jnp.array([0.01])
    ),
    resolution=50,
    threshold=BootstrapThresholdConfig(confidence_level=0.05, n_bootstrap=250, single_beta=False),
    seed=0
)
