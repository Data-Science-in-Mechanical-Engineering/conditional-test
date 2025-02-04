from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
from jax import numpy as jnp

from src.config import ThresholdConfig, DataConfig, IIDDataConfig, SpaceConfig, GaussianMixtureNoiseConfig, \
    KernelParametrizationConfig, BootstrapThresholdConfig
from src.data import RKHSFnSampling
from src.random import uniform_rkhs_subspace_fn, uniform_rkhs_subspace_fns
from src.rkhs import Kernel, RKHSFn


@dataclass(frozen=True)
class BaseConfig(ABC):
    kernel: KernelParametrizationConfig  # configuration of the kernel
    threshold: ThresholdConfig  # configuration for the computation of the decision threshold on the CMMD
    data: DataConfig  # configuration for the dataset
    n_functions: int  # number of RKHS functions to average over
    n_basis_points: int  # number of points on which each function is centered
    n_datasets: int  # number of datasets to sample for each function
    n_evaluation_points: int  # number of points at which the CMMD is evaluated
    rkhs_ball_radius: float  # radius of the RKHS ball from which the functions are sampled
    resolution_power: int  # power of the resolution grid
    n_bootstrap_rate: int  # number of bootstrap samples to bootstrap the distribution *of the rejection rates*
    seed: int  # random seed

    def __post_init__(self):
        assert self.n_basis_points >= 1
        assert self.n_datasets >= 1
        assert self.rkhs_ball_radius > 0
        assert self.resolution_power >= 1

    def confidence_levels(self) -> jnp.ndarray:
        return jnp.linspace(0, self.threshold.confidence_level, self.resolution_power + 1)[1:]

    @abstractmethod
    def sample_rkhs_fn_pair(self, kernel: Kernel, key: jax.Array) -> tuple[RKHSFn, RKHSFn]:
        raise NotImplementedError

    @abstractmethod
    def sample_dataset_pair(
            self,
            kernel: Kernel,
            fn_1: RKHSFn, fn_2: RKHSFn,
            key: jax.Array
    ) -> tuple[RKHSFnSampling, RKHSFnSampling]:
        raise NotImplementedError

    @abstractmethod
    def sub_gaussian_std(self) -> float:
        raise NotImplementedError

    def sample_rkhs_fn(
            self,
            kernel: Kernel,
            key: jax.Array,
            radius: float | None = None,
            sphere: bool = False
    ) -> RKHSFn:
        if radius is None:
            radius = self.rkhs_ball_radius

        key_1, key_2 = jax.random.split(key)

        return uniform_rkhs_subspace_fns(
            kernel=kernel,
            xs_batch=self.data.space.uniform_sample((self.n_functions, self.n_basis_points), key_1),
            radius=radius,
            sphere=sphere,
            key=key_2
        )

    def sample_function_evaluations(self, kernel: Kernel, fns: RKHSFn, key: jax.Array) -> RKHSFnSampling:
        @partial(jax.vmap)
        def batch_sampling_fn(key_: jax.Array) -> jnp.ndarray:
            return self.data.sample(key_)

        @partial(jax.vmap)
        def batch_evaluation_fn(fn: RKHSFn, xs_: jnp.ndarray) -> jnp.ndarray:
            return kernel.evaluate.one_many(fn, xs_)

        xs = batch_sampling_fn(jax.random.split(key, self.n_functions))
        ys = batch_evaluation_fn(fns, xs)

        return RKHSFnSampling(xs=xs, ys=ys)

    def sample_noisy_datasets(
            self,
            kernel: Kernel, functions: RKHSFn,
            noise_distribution: GaussianMixtureNoiseConfig,
            key: jax.Array
    ) -> RKHSFnSampling:
        @partial(jax.vmap)
        def batch_fn(key_: jax.Array) -> RKHSFnSampling:
            key_data, key_noise = jax.random.split(key_)

            evaluations = self.sample_function_evaluations(kernel, functions, key_data)

            return RKHSFnSampling(
                xs=evaluations.xs,
                ys=evaluations.ys + noise_distribution.sample(evaluations.ys.shape, key_noise)
            )

        keys = jax.random.split(key, self.n_datasets)

        datasets = batch_fn(keys)

        return RKHSFnSampling(
            xs=datasets.xs.reshape((self.n_functions * self.n_datasets, -1, self.data.space.dim)),
            ys=datasets.ys.reshape((self.n_functions * self.n_datasets, -1))
        )


@dataclass(frozen=True)
class SingleFnConfig(BaseConfig, ABC):
    def sample_rkhs_fn_pair(self, kernel: Kernel, key: jax.Array) -> tuple[RKHSFn, RKHSFn]:
        fn = self.sample_rkhs_fn(kernel, key)
        return fn, fn


class DifferentFnConfig(BaseConfig, ABC):
    def sample_rkhs_fn_pair(self, kernel: Kernel, key: jax.Array) -> tuple[RKHSFn, RKHSFn]:
        key_1, key_2 = jax.random.split(key)

        fn_1 = self.sample_rkhs_fn(kernel, key_1)
        fn_2 = self.sample_rkhs_fn(kernel, key_2)

        return fn_1, fn_2


@dataclass(frozen=True)
class SingleNoiseConfig(BaseConfig, ABC):
    noise: GaussianMixtureNoiseConfig  # noise configuration for the dataset

    def sample_dataset_pair(
            self,
            kernel: Kernel,
            fn_1: RKHSFn, fn_2: RKHSFn,
            key: jax.Array
    ) -> tuple[RKHSFnSampling, RKHSFnSampling]:
        key_1, key_2 = jax.random.split(key)

        dataset_1 = self.sample_noisy_datasets(kernel, fn_1, self.noise, key_1)
        dataset_2 = self.sample_noisy_datasets(kernel, fn_2, self.noise, key_2)

        return dataset_1, dataset_2

    def sub_gaussian_std(self) -> float:
        return float(self.noise.std.max())


@dataclass(frozen=True)
class DifferentNoiseConfig(BaseConfig, ABC):
    noise_1: GaussianMixtureNoiseConfig  # noise configuration for the first dataset
    noise_2: GaussianMixtureNoiseConfig  # noise configuration for the second dataset

    def sample_dataset_pair(
            self,
            kernel: Kernel,
            fn_1: RKHSFn, fn_2: RKHSFn,
            key: jax.Array
    ) -> tuple[RKHSFnSampling, RKHSFnSampling]:
        key_1, key_2 = jax.random.split(key)

        dataset_1 = self.sample_noisy_datasets(kernel, fn_1, self.noise_1, key_1)
        dataset_2 = self.sample_noisy_datasets(kernel, fn_2, self.noise_2, key_2)

        return dataset_1, dataset_2

    def sub_gaussian_std(self) -> float:
        return max(
            float(self.noise_1.std.max()),
            float(self.noise_2.std.max())
        )


@dataclass(frozen=True)
class SingleFnSingleNoiseConfig(SingleFnConfig, SingleNoiseConfig):
    pass


@dataclass(frozen=True)
class SingleFnDifferentNoiseConfig(SingleFnConfig, DifferentNoiseConfig):
    pass


@dataclass(frozen=True)
class DifferentFnSingleNoiseConfig(DifferentFnConfig, SingleNoiseConfig):
    pass


@dataclass(frozen=True)
class DifferentFnDifferentNoiseConfig(DifferentFnConfig, DifferentNoiseConfig):
    pass


@dataclass(frozen=True)
class DisturbedFnSingleNoiseConfig(SingleNoiseConfig):
    disturbance: float  # disturbance level

    def sample_rkhs_fn_pair(self, kernel: Kernel, key: jax.Array) -> tuple[RKHSFn, RKHSFn]:
        key_1, key_2 = jax.random.split(key)

        fn_1 = self.sample_rkhs_fn(kernel, key_1)
        disturbance = self.sample_rkhs_fn(kernel, key_2, radius=self.disturbance, sphere=True)

        # fn_2 = fn_1 + disturbance (batched addition)
        fn_2 = RKHSFn(
            coefficients=jnp.concatenate([fn_1.coefficients, disturbance.coefficients], axis=-1),
            points=jnp.concatenate([fn_1.points, disturbance.points], axis=-2)
        )

        return fn_1, fn_2


class PositiveRate(NamedTuple):
    uniform: jnp.ndarray
    local: jnp.ndarray
    bootstrap_uniform_distribution: jnp.ndarray
    bootstrap_local_distribution: jnp.ndarray


DEFAULT_ARGS = dict(
    n_functions=10,
    n_basis_points=20,
    n_datasets=100,
    n_evaluation_points=100,
    rkhs_ball_radius=1.0,
    resolution_power=100,
    n_bootstrap_rate=100,
    seed=0
)

DEFAULT_MAX_CONFIDENCE_LEVEL = 1

DEFAULT_BOOTSTRAP = BootstrapThresholdConfig(
    confidence_level=DEFAULT_MAX_CONFIDENCE_LEVEL,
    n_bootstrap=100,
    single_beta=False
)

DEFAULT_SPACE = SpaceConfig(dim=2, min=-1, max=1)

DEFAULT_IID_DATA = IIDDataConfig(
    space=DEFAULT_SPACE,
    dataset_size=100
)
