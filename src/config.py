from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from numpy.random.mtrand import Sequence

from src.bounds import analytical_cmmd_thresholds, bootstrap_cmmd_thresholds
from src.data import collect_rotation_trajectories
from src.rkhs import Kernel, CKME, GaussianKernel, RKHSFn, LinearKernel
from src.rkhs.kernels import PolynomialKernel
from src.util import KernelParametrization


@dataclass(frozen=True)
class KernelConfig(ABC):
    ndim: int

    @abstractmethod
    def make(self) -> Kernel:
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianKernelConfig(KernelConfig):
    bandwidth: float

    def make(self) -> GaussianKernel:
        return GaussianKernel(bandwidth=self.bandwidth, ndim=self.ndim)


@dataclass(frozen=True)
class PolynomialKernelConfig(KernelConfig):
    degree: int

    def make(self) -> PolynomialKernel:
        return PolynomialKernel(degree=self.degree, ndim=self.ndim)


@dataclass(frozen=True)
class LinearKernelConfig(KernelConfig):
    def make(self) -> LinearKernel:
        return LinearKernel(ndim=self.ndim)


@dataclass(frozen=True)
class KernelParametrizationConfig[T1: Kernel, T2: Kernel]:
    x_config: KernelConfig  # kernel config for the input space
    y_config: KernelConfig  # kernel config for the output space
    regularization: float  # regularization parameter for the kernel ridge regression

    def make(self) -> KernelParametrization[T1, T2]:
        return KernelParametrization(
            x=self.x_config.make(),
            y=self.y_config.make(),
            regularization=self.regularization
        )

    def __post_init__(self):
        assert self.regularization > 0, "Regularization parameter must be positive."


@dataclass(frozen=True)
class SpaceConfig:
    dim: int  # dimension of the space
    min: jnp.ndarray | float  # lower boundaries of the space
    max: jnp.ndarray | float  # upper boundaries of the space

    @property
    def max_vector(self):
        if isinstance(self.min, float | int):
            return jnp.full(self.dim, self.max).astype(jnp.float32)
        return self.max

    @property
    def min_vector(self):
        if isinstance(self.min, float | int):
            return jnp.full(self.dim, self.min).astype(jnp.float32)
        return self.min

    def __post_init__(self):
        assert self.dim >= 1, "Dimension of the space must be positive."
        assert self.min_vector.shape == (self.dim,), "Lower boundaries must have the same dimension as the space."
        assert self.max_vector.shape == (self.dim,), "Upper boundaries must have the same dimension as the space."
        assert jnp.all(self.min <= self.max), "Lower boundaries must be less than or equal to upper boundaries."

    def uniform_sample(self, shape: tuple[int, ...] | int, key: jax.Array) -> jnp.ndarray:
        if isinstance(shape, int):
            shape = (shape,)

        return jax.random.uniform(key, shape=shape + (self.dim,), minval=self.min, maxval=self.max)

    def normal_sample(self, shape: tuple[int, ...] | int, key: jax.Array) -> jnp.ndarray:
        if isinstance(shape, int):
            shape = (shape,)

        return jax.random.normal(key, shape=shape + (self.dim,))

    def discretization(self, resolution: int) -> jnp.ndarray:
        lin_spaces = [jnp.linspace(lo, hi, resolution) for lo, hi in zip(self.min_vector, self.max_vector)]
        grids = jnp.meshgrid(*lin_spaces, indexing="ij")
        return jnp.stack(grids, axis=-1).reshape((-1, self.dim))


@dataclass(frozen=True)
class DataConfig(ABC):
    space: SpaceConfig  # configuration of the input space

    @abstractmethod
    def sample(self, key: jax.Array) -> jnp.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class IIDDataConfig(DataConfig):
    dataset_size: int  # number of samples in the dataset

    def __post_init__(self):
        assert self.dataset_size >= 1

    def sample(self, key: jax.Array) -> jnp.ndarray:
        return self.space.uniform_sample(self.dataset_size, key)


@dataclass(frozen=True)
class RotationDataConfig(DataConfig):
    n_initializations: int  # number of independent initializations for each dataset
    trajectory_length: int  # number of steps in the trajectory
    n_rotations: float  # number of rotations around the origin for each trajectory
    damping_factor: float  # damping factor of the rotation

    @property
    def dataset_size(self) -> int:
        return self.n_initializations * self.trajectory_length

    def __post_init__(self):
        assert self.n_initializations >= 1
        assert self.trajectory_length >= 1
        assert self.n_rotations > 0
        assert self.space.dim == 2

    def sample(self, key: jax.Array) -> jnp.ndarray:
        return collect_rotation_trajectories(
            xs_init=self.space.uniform_sample(self.n_initializations, key),
            n_rotations=self.n_rotations,
            n_samples=self.trajectory_length,
            damping_factor=self.damping_factor
        ).reshape(-1, self.space.dim)


@dataclass(frozen=True)
class ThresholdConfig(ABC):
    confidence_level: float  # confidence level for which to compute the thresholds

    def __post_init__(self):
        assert 0 < self.confidence_level <= 1, "Confidence level must be in (0, 1]."

    @abstractmethod
    def thresholds(
            self,
            kernel: KernelParametrization,
            ckme_1: CKME, ckme_2: CKME,
            es: jnp.ndarray,
            rkhs_norm_1: float, rkhs_norm_2: float,
            sub_gaussian_std_1: float, sub_gaussian_std_2: float,
            power: float | jnp.ndarray,
            key: jax.Array
    ) -> jnp.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class AnalyticalThresholdConfig(ThresholdConfig):
    def thresholds(
            self,
            kernel: KernelParametrization,
            ckme_1: CKME, ckme_2: CKME,
            es: jnp.ndarray,
            rkhs_norm_1: float, rkhs_norm_2: float,
            sub_gaussian_std_1: float, sub_gaussian_std_2: float,
            power: float | jnp.ndarray,
            key: jax.Array
    ) -> jnp.ndarray:
        return analytical_cmmd_thresholds(
            kernel=kernel.x,
            ckme_1=ckme_1, ckme_2=ckme_2,
            es_1=es, es_2=es,
            rkhs_norm_1=rkhs_norm_1, rkhs_norm_2=rkhs_norm_2,
            sub_gaussian_std_1=sub_gaussian_std_1, sub_gaussian_std_2=sub_gaussian_std_2,
            power=power
        )


@dataclass(frozen=True)
class BootstrapThresholdConfig(ThresholdConfig):
    single_beta: bool  # whether to bootstrap the two beta terms separately or a single beta
    n_bootstrap: int  # number of bootstrapped datasets to estimate the distribution of the threshold under H_0

    def __post_init__(self):
        super().__post_init__()
        assert self.n_bootstrap >= 1, "Number of bootstrap datasets must be positive."

    def thresholds(
            self,
            kernel: KernelParametrization,
            ckme_1: CKME, ckme_2: CKME,
            es: jnp.ndarray,
            rkhs_norm_1: float, rkhs_norm_2: float,
            sub_gaussian_std_1: float, sub_gaussian_std_2: float,
            power: float | jnp.ndarray,
            key: jax.Array
    ) -> jnp.ndarray:
        return bootstrap_cmmd_thresholds(
            kernel_x=kernel.x, kernel_y=kernel.y,
            n_bootstrap=self.n_bootstrap,
            ckme_1=ckme_1, ckme_2=ckme_2,
            es_1=es, es_2=es,
            power=power,
            single_beta=self.single_beta,
            key=key
        )


@dataclass(frozen=True)
class NoiseConfig(ABC):
    @abstractmethod
    def sample(self, shape: Sequence[int] | int, key: jax.Array) -> jnp.ndarray:
        raise NotImplementedError


class SubGaussianNoiseConfig(NoiseConfig, ABC):
    @abstractmethod
    def sub_gaussian_std(self) -> float:
        raise NotImplementedError


class NoiseWithGaussianCMMDClosedForm(ABC):
    @classmethod
    def closed_form_mmd(
            cls,
            distribution_1: Self, distribution_2: Self,
            kernel_x: Kernel, kernel_y: GaussianKernel,
            fn_1: RKHSFn, fn_2: RKHSFn,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        ...

    @classmethod
    def closed_form_mmd_batch(
            cls,
            noise_1: Self, noise_2: Self,
            kernel_x: Kernel, kernel_y: GaussianKernel,
            fn_1: RKHSFn, fn_2: RKHSFn,
            xs: jnp.ndarray
    ) -> jnp.ndarray:
        @partial(jax.vmap)
        def batch_fn(x: jnp.ndarray) -> jnp.ndarray:
            return cls.closed_form_mmd(
                distribution_1=noise_1, distribution_2=noise_2,
                kernel_x=kernel_x, kernel_y=kernel_y,
                fn_1=fn_1, fn_2=fn_2,
                x=x
            )

        return batch_fn(xs)


@dataclass(frozen=True)
class GaussianNoiseConfig(SubGaussianNoiseConfig, NoiseWithGaussianCMMDClosedForm):
    mean: jnp.ndarray  # mean of the Gaussian noise
    std: jnp.ndarray  # standard deviation of the Gaussian noise

    def __post_init__(self):
        assert self.mean.ndim == 1
        assert self.std.ndim == 1
        assert self.mean.shape == self.std.shape
        assert jnp.all(self.std >= 0)

    @classmethod
    def gaussian_kernel_delta(
            cls,
            mean_1: jnp.ndarray | float, mean_2: jnp.ndarray | float,
            variance_1: jnp.ndarray, variance_2: jnp.ndarray,
            kernel: GaussianKernel
    ) -> jnp.ndarray:
        mean = mean_1 - mean_2
        variance = variance_1 + variance_2

        a = jnp.prod(1 / jnp.sqrt((1 + variance / kernel.bandwidth)))
        b = jnp.exp(-1 / 2 * mean ** 2 / (variance + kernel.bandwidth)).prod()
        return a * b

    @classmethod
    def closed_form_mmd(
            cls,
            distribution_1: Self, distribution_2: Self,
            kernel_x: Kernel, kernel_y: GaussianKernel,
            fn_1: RKHSFn, fn_2: RKHSFn,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        assert x.ndim == kernel_x.ndim

        mean_1 = kernel_x.evaluate(fn_1, x) + distribution_1.mean
        mean_2 = kernel_x.evaluate(fn_2, x) + distribution_2.mean

        var_1 = distribution_1.std ** 2
        var_2 = distribution_2.std ** 2

        squared_mmd = (
                GaussianNoiseConfig.gaussian_kernel_delta(0, 0, var_1, var_1, kernel_y)
                + GaussianNoiseConfig.gaussian_kernel_delta(0, 0, var_2, var_2, kernel_y)
                - 2 * GaussianNoiseConfig.gaussian_kernel_delta(mean_1, mean_2, var_1, var_2, kernel_y)
        )

        return jnp.sqrt(squared_mmd)

    def sub_gaussian_std(self) -> float:
        return float(self.std.max())

    def sample(self, shape: Sequence[int] | int, key: jax.Array) -> jnp.ndarray:
        if isinstance(shape, int):
            shape = (shape,)

        return jax.random.normal(key, shape=shape) * self.std + self.mean

    def __hash__(self):
        return hash((tuple(self.mean.tolist()), tuple(self.std.tolist())))


@dataclass(frozen=True)
class GaussianMixtureNoiseConfig(SubGaussianNoiseConfig, NoiseWithGaussianCMMDClosedForm):
    mean: jnp.ndarray  # means of the Gaussian components
    std: jnp.ndarray  # standard deviations of the Gaussian components

    @property
    def n(self) -> int:
        return self.mean.shape[0]

    def __post_init__(self):
        assert self.mean.shape == self.std.shape

    def sub_gaussian_std(self) -> float:
        if self.n == 1:
            return float(self.std[0])
        else:
            return -1

    def sample(self, shape: Sequence[int] | int, key: jax.Array) -> jnp.ndarray:
        if isinstance(shape, int):
            shape = (shape,)

        sample_shape = shape + (self.mean.shape[1:])

        key_1, key_2 = jax.random.split(key)

        components = jax.random.randint(key_1, shape=shape, minval=0, maxval=self.n)

        return jax.random.normal(key_2, shape=sample_shape) * self.std[components] + self.mean[components]

    @classmethod
    def gaussian_mixture_kernel_delta(
            cls,
            means_1: jnp.ndarray, means_2: jnp.ndarray,
            stds_1: jnp.ndarray, stds_2: jnp.ndarray,
            kernel: GaussianKernel
    ):
        @partial(jax.vmap, in_axes=(0, None, 0, None))
        @partial(jax.vmap, in_axes=(None, 0, None, 0))
        def pairwise_fn(mean_1: jnp.ndarray, mean_2: jnp.ndarray, std_1: jnp.ndarray, std_2: jnp.ndarray):
            return GaussianNoiseConfig.gaussian_kernel_delta(mean_1, mean_2, std_1, std_2, kernel)

        return pairwise_fn(means_1, means_2, stds_1, stds_2).mean()

    @classmethod
    def closed_form_mmd(
            cls,
            distribution_1: Self, distribution_2: Self,
            kernel_x: Kernel, kernel_y: GaussianKernel,
            fn_1: RKHSFn, fn_2: RKHSFn,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        assert x.ndim == kernel_x.ndim

        means_1 = kernel_x.evaluate(fn_1, x) + distribution_1.mean
        means_2 = kernel_x.evaluate(fn_2, x) + distribution_2.mean

        vars_1 = distribution_1.std ** 2
        vars_2 = distribution_2.std ** 2

        dp_11 = GaussianMixtureNoiseConfig.gaussian_mixture_kernel_delta(means_1, means_1, vars_1, vars_1, kernel_y)
        dp_22 = GaussianMixtureNoiseConfig.gaussian_mixture_kernel_delta(means_2, means_2, vars_2, vars_2, kernel_y)
        dp_12 = GaussianMixtureNoiseConfig.gaussian_mixture_kernel_delta(means_1, means_2, vars_1, vars_2, kernel_y)

        squared_mmd = dp_11 + dp_22 - 2 * dp_12

        return jnp.sqrt(squared_mmd)

    def __hash__(self):
        return hash((tuple(self.mean.tolist()), tuple(self.std.tolist())))
