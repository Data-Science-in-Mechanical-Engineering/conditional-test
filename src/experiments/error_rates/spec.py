from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Self, Iterator

import jax
import jax.numpy as jnp
from jax import Array

from src.data import RKHSFnSampling
from src.experiments.error_rates.tests import BootstrapTest, AnalyticalTest, HuLeiTest
from src.rkhs import RKHSFn, Kernel, GaussianKernel
from src.spec import NoiseSpec, VectorKernelSpec, RKHSFnSpec, SpaceSpec


# rejection sampling algorithm for sampling uniformly from a hypercube without a centered ball
def _sample_uniform_without_ball(space: SpaceSpec, radius: float, shape: tuple[int, ...], key: Array) -> Array:
    assert (2 * radius <= space.max_vector - space.min_vector).all()
    center = (space.max_vector - space.min_vector) / 2 + space.min_vector

    def get_violations(points: Array) -> Array:
        return jnp.linalg.norm(points - center, axis=-1, keepdims=True) <= radius

    key, key_sample = jax.random.split(key)
    sample = space.uniform_sample(shape=shape, key=key)
    violations = get_violations(sample)

    while violations.any():
        key, key_sample = jax.random.split(key)
        corrections = space.uniform_sample(shape=shape, key=key)
        sample = jnp.where(violations, corrections, sample)
        violations = get_violations(sample)

    return sample


# sample uniformly from a hyperball with a specified radius
def _sample_uniform_ball(radius: float, dim: int, shape: tuple[int, ...], key: Array) -> Array:
    direction = jax.random.normal(key, shape=shape + (dim,))
    scaling = jax.random.uniform(key, shape=shape + (1,)) ** (1 / dim)
    return direction / jnp.linalg.norm(direction, axis=-1, keepdims=True) * radius * scaling


class RKHSFnSamplingPair(NamedTuple):
    fn_1: RKHSFn
    fn_2: RKHSFn
    dataset_1: RKHSFnSampling
    dataset_2: RKHSFnSampling

    @classmethod
    def sample(
            cls,
            fn_1: RKHSFn, fn_2: RKHSFn,
            noise_1: NoiseSpec, noise_2: NoiseSpec,
            space: SpaceSpec,
            dataset_size: int, n_datasets: int,
            key: Array
    ) -> Self:
        key, key_xs_1, key_xs_2 = jax.random.split(key, 3)
        xs_1 = space.uniform_sample(shape=(n_datasets, dataset_size), key=key_xs_1)
        xs_2 = space.uniform_sample(shape=(n_datasets, dataset_size), key=key_xs_2)
        return cls.sample_at(fn_1, fn_2, xs_1, xs_2, noise_1, noise_2, key)

    @classmethod
    def sample_at(
            cls,
            fn_1: RKHSFn, fn_2: RKHSFn,
            xs_1: Array, xs_2: Array,
            noise_1: NoiseSpec, noise_2: NoiseSpec,
            key: Array
    ) -> Self:
        sampling_1 = RKHSFnSampling.at_points(fn_1, xs_1)
        sampling_2 = RKHSFnSampling.at_points(fn_2, xs_2)

        key_noise_1, key_noise_2 = jax.random.split(key)

        noise_1 = noise_1.sample(sampling_1.ys.shape, key_noise_1)
        noise_2 = noise_2.sample(sampling_2.ys.shape, key_noise_2)

        noisy_sampling_1 = sampling_1.add_y(noise_1)
        noisy_sampling_2 = sampling_2.add_y(noise_2)

        return RKHSFnSamplingPair(fn_1, fn_2, noisy_sampling_1, noisy_sampling_2)

    def __len__(self) -> int:
        assert len(self.dataset_1) == len(self.dataset_2)
        return len(self.dataset_1)

    def __getitem__(self, item) -> Self:
        return RKHSFnSamplingPair(self.fn_1, self.fn_2, self.dataset_1[item], self.dataset_2[item])

    def __iter__(self) -> Iterator[Self]:
        for i in range(len(self)):
            yield self[i]


@dataclass(frozen=True)
class _DatasetPairSamplingSpec:
    rkhs_fn: RKHSFnSpec
    noise_1: NoiseSpec
    noise_2: NoiseSpec
    dataset_size: int

    def __post_init__(self):
        assert self.dataset_size >= 1

    @abstractmethod
    def sample(self, kernel: Kernel, space: SpaceSpec, n_datasets: int, key: Array) -> RKHSFnSamplingPair:
        raise NotImplementedError


@dataclass(frozen=True)
class DatasetPairSamplingSpec(_DatasetPairSamplingSpec):
    single_fn: bool

    def sample(self, kernel: Kernel, space: SpaceSpec, n_datasets: int, key: Array) -> RKHSFnSamplingPair:
        key, key_fn_1, key_fn_2 = jax.random.split(key, 3)
        rkhs_fn_1 = self.rkhs_fn.sample(kernel, space, key=key_fn_1, surface=True)

        if self.single_fn:
            rkhs_fn_2 = rkhs_fn_1
        else:
            rkhs_fn_2 = self.rkhs_fn.sample(kernel, space, key=key_fn_2, surface=True)

        return RKHSFnSamplingPair.sample(
            fn_1=rkhs_fn_1, fn_2=rkhs_fn_2,
            noise_1=self.noise_1, noise_2=self.noise_2,
            space=space,
            dataset_size=self.dataset_size, n_datasets=n_datasets,
            key=key
        )


@dataclass(frozen=True)
class DisturbedDatasetPairSamplingSpec(_DatasetPairSamplingSpec):
    n_disturbance_basis_point: int
    relative_disturbance_norm: float

    @property
    def disturbance_norm(self) -> float:
        return self.rkhs_fn.ball_radius * self.relative_disturbance_norm

    def __post_init__(self):
        super().__post_init__()
        assert self.n_disturbance_basis_point >= 1

    @property
    def disturbance_rkhs_fn(self) -> RKHSFnSpec:
        return RKHSFnSpec(n_basis_points=self.n_disturbance_basis_point, ball_radius=self.disturbance_norm)

    def sample(self, kernel: Kernel, space: SpaceSpec, n_datasets: int, key: Array) -> RKHSFnSamplingPair:
        key, key_fn, key_disturbance = jax.random.split(key, 3)

        rkhs_fn = self.rkhs_fn.sample(kernel, space, key=key_fn, surface=True)
        disturbance_fn = self.disturbance_rkhs_fn.sample(kernel, space, key=key, surface=True)
        rkhs_fn_disturbed = rkhs_fn + disturbance_fn

        return RKHSFnSamplingPair.sample(
            fn_1=rkhs_fn, fn_2=rkhs_fn_disturbed,
            noise_1=self.noise_1, noise_2=self.noise_2,
            space=space,
            dataset_size=self.dataset_size, n_datasets=n_datasets,
            key=key
        )


@dataclass(frozen=True)
class WeightedDisturbedDatasetPairSamplingSpec(_DatasetPairSamplingSpec):
    relative_disturbance_norm: float
    tolerance: float
    weight: float

    @property
    def disturbance_norm(self) -> float:
        return self.rkhs_fn.ball_radius * self.relative_disturbance_norm

    def __post_init__(self):
        super().__post_init__()
        assert self.relative_disturbance_norm >= 0
        assert self.tolerance >= 0
        assert self.weight >= 0
        assert self.weight <= 1

    def _sample_state_space(self, radius: float, space: SpaceSpec, n_datasets: int, key: Array) -> Array:
        key_inner, key_outer, key_combined = jax.random.split(key, 3)

        inner_sample = _sample_uniform_ball(
            radius, dim=space.dim, shape=(n_datasets, self.dataset_size), key=key_inner
        )

        outer_sample = _sample_uniform_without_ball(
            space, radius=radius, shape=(n_datasets, self.dataset_size), key=key_outer
        )

        indicator = jax.random.uniform(key_combined, shape=(n_datasets, self.dataset_size, 1)) < self.weight

        return indicator * inner_sample + (1 - indicator) * outer_sample

    def sample(self, kernel: GaussianKernel, space: SpaceSpec, n_datasets: int, key: Array) -> RKHSFnSamplingPair:
        center = (space.max_vector - space.min_vector) / 2 + space.min_vector

        disturbance_fn = RKHSFn(
            kernel=kernel,
            coefficients=jnp.array([self.disturbance_norm * kernel(center, center)]),
            points=center[None]
        )

        radius = jnp.sqrt(2 * self.disturbance_norm * kernel.bandwidth * jnp.log(1 / self.tolerance)).item()

        key, key_fn = jax.random.split(key)
        rkhs_fn = self.rkhs_fn.sample(kernel, space, key=key_fn, surface=True)
        rkhs_fn_disturbed = rkhs_fn + disturbance_fn

        key, key_xs_1, key_xs_2 = jax.random.split(key, 3)
        xs_1 = self._sample_state_space(radius, space, n_datasets, key_xs_1)
        xs_2 = self._sample_state_space(radius, space, n_datasets, key_xs_2)

        return RKHSFnSamplingPair.sample_at(
            fn_1=rkhs_fn, fn_2=rkhs_fn_disturbed,
            xs_1=xs_1, xs_2=xs_2,
            noise_1=self.noise_1, noise_2=self.noise_2,
            key=key
        )


@dataclass
class Config:
    kernel: VectorKernelSpec
    space: SpaceSpec
    datasets: _DatasetPairSamplingSpec
    test: AnalyticalTest.Spec | BootstrapTest.Spec | HuLeiTest.Spec
    resolution: int
    n_repetitions: int
    seed: int

    @property
    def rkhs_fn(self) -> RKHSFnSpec:
        return RKHSFnSpec(n_basis_points=12, ball_radius=1.0)

    @property
    def significance_levels(self) -> Array:
        return jnp.linspace(0, 1, 100).at[0].set(0.001)

    def __post_init__(self):
        assert self.resolution >= 1
        assert self.n_repetitions >= 1

    def sample_dataset_pairs(self, kernel: Kernel, key: Array) -> RKHSFnSamplingPair:
        return self.datasets.sample(kernel, self.space, self.n_repetitions, key)
