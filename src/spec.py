from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from src.data import collect_rotation
from src.rkhs import Kernel, GaussianKernel, LinearKernel, PolynomialKernel, VectorKernel, RKHSFn
from src.rkhs.sampling import uniform_rkhs_subspace_ball


@dataclass(frozen=True)
class RKHSFnSpec:
    n_basis_points: int  # subspace dimension of the RKHS
    ball_radius: float  # radius of the centered ball in the RKHS subspace

    def __post_init__(self):
        assert self.n_basis_points >= 1
        assert self.ball_radius >= 0

    def sample(
            self, kernel: Kernel, space: SpaceSpec, key: Array, surface: bool = False, shape: Sequence[int] = ()
    ) -> RKHSFn:
        key_points, key_fn = jax.random.split(key)

        shape = tuple(shape)
        basis_points = space.uniform_sample(shape + (self.n_basis_points,), key_points)

        return uniform_rkhs_subspace_ball(key_fn, basis_points, kernel, self.ball_radius, surface)


@dataclass(frozen=True)
class KernelSpec(ABC):
    @abstractmethod
    def make(self) -> Kernel:
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianKernelSpec(KernelSpec):
    bandwidth: float  # bandwidth of the Gaussian kernel
    ndim: int  # number of axes of the data

    def __post_init__(self):
        assert self.bandwidth > 0
        assert self.ndim >= 0

    def make(self) -> Kernel:
        return GaussianKernel(self.bandwidth, self.ndim)


@dataclass(frozen=True)
class LinearKernelSpec(KernelSpec):
    ndim: int  # number of axes of the data

    def __post_init__(self):
        assert self.ndim >= 0

    def make(self) -> Kernel:
        return LinearKernel(self.ndim)


@dataclass(frozen=True)
class PolynomialKernelSpec(KernelSpec):
    degree: int  # degree of the polynomial kernel
    ndim: int  # number of axes of the data

    def __post_init__(self):
        assert self.degree >= 0
        assert self.ndim >= 0

    def make(self) -> Kernel:
        return PolynomialKernel(self.degree, self.ndim)


@dataclass(frozen=True)
class VectorKernelSpec:
    x: KernelSpec  # specification of the kernel on the input space
    y: KernelSpec  # specification of the kernel on the output space
    regularization: float  # regularization for the kernel ridge regression

    def __post_init__(self):
        assert self.regularization > 0

    def make(self) -> VectorKernel:
        return VectorKernel(
            x=self.x.make(),
            y=self.y.make(),
            regularization=self.regularization
        )


@dataclass(frozen=True)
class SpaceSpec:
    dim: int  # dimension of the space
    min: Array | float  # lower boundaries of the space
    max: Array | float  # upper boundaries of the space

    @property
    def max_vector(self) -> Array:
        max_vector = jnp.asarray(self.max)

        if jnp.isscalar(max_vector):
            max_vector = jnp.full(self.dim, max_vector, dtype=jnp.float32)

        return max_vector

    @property
    def min_vector(self) -> Array:
        min_vector = jnp.asarray(self.min)

        if jnp.isscalar(min_vector):
            min_vector = jnp.full(self.dim, min_vector, dtype=jnp.float32)

        return min_vector

    def __post_init__(self):
        assert self.dim >= 1
        assert self.min_vector.shape == (self.dim,)
        assert self.max_vector.shape == (self.dim,)
        assert jnp.all(self.min <= self.max)

    def uniform_sample(self, shape: Sequence[int] | int, key: Array) -> Array:
        if isinstance(shape, int):
            shape = (shape,)

        shape = tuple(shape) + (self.dim,)

        return jax.random.uniform(key, shape=shape, minval=self.min_vector, maxval=self.max_vector)

    def discretization(self, resolution: int) -> Array:
        lin_spaces = [jnp.linspace(lo, hi, resolution) for lo, hi in zip(self.min_vector, self.max_vector)]
        grids = jnp.meshgrid(*lin_spaces, indexing="ij")
        return jnp.stack(grids, axis=-1).reshape((-1, self.dim))


@dataclass(frozen=True)
class SamplingSpec(ABC):
    @abstractmethod
    def sample(self, space: SpaceSpec, shape: Sequence[int] | int, key: Array) -> Array:
        raise NotImplementedError


@dataclass(frozen=True)
class UniformSamplingSpec(SamplingSpec):
    dataset_size: int  # number of samples in the dataset

    def sample(self, space: SpaceSpec, shape: Sequence[int] | int, key: Array) -> Array:
        if isinstance(shape, int):
            shape = (shape,)

        return space.uniform_sample(tuple(shape) + (self.dataset_size,), key)


@dataclass(frozen=True)
class RotationSamplingSpec(SamplingSpec):
    trajectory_length: int  # number of samples per trajectory
    n_rotations: float  # number of rotations per trajectory
    damping_factor: float  # damping of the rotation
    n_initializations: int  # number of independent initializations

    def sample(self, space: SpaceSpec, shape: Sequence[int] | int, key: Array) -> Array:
        assert space.dim == 2

        if isinstance(shape, int):
            shape = (shape,)

        xs_init = space.uniform_sample((self.n_initializations,) + shape, key)

        return self.sample_from_initialization(xs_init)

    def sample_from_initialization(self, xs_init: Array) -> Array:
        assert xs_init.shape[-1] == 2

        xs_init = collect_rotation(xs_init, self.n_rotations, self.trajectory_length, self.damping_factor)

        return xs_init.reshape(-1, *xs_init.shape[2:-1], 2)


@dataclass(frozen=True)
class MultiSignificanceTestSpec:
    min: float  # smallest significance level of the two-sample test
    max: float  # largest significance level of the two-sample test
    n_levels: int  # number of significance levels

    def __post_init__(self):
        assert self.min > 0
        assert self.min <= self.max
        assert self.max <= 1
        assert self.n_levels >= 1

    def levels(self) -> Array:
        return jnp.linspace(self.min, self.max, self.n_levels)


@dataclass(frozen=True)
class TestSpec[T: float | MultiSignificanceTestSpec](ABC):
    significance_level: T  # significance level of the two-sample test

    def significance_level_array(self) -> ArrayLike:
        if isinstance(self.significance_level, float):
            return jnp.asarray(self.significance_level)
        else:
            return self.significance_level.levels()

    def __post_init__(self):
        if isinstance(self.significance_level, float):
            assert 0 < self.significance_level < 1


@dataclass(frozen=True)
class AnalyticalTestSpec(TestSpec):
    pass


@dataclass(frozen=True)
class BootstrapTestSpec[T: float | MultiSignificanceTestSpec](TestSpec[T]):
    n_bootstrap: int  # number of bootstrap samples


@dataclass(frozen=True)
class NoiseSpec(ABC):
    @abstractmethod
    def sample(self, shape: Sequence[int] | int, key: jax.Array) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sub_gaussian_std(self) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianNoiseSpec(NoiseSpec):
    mean: jnp.ndarray  # mean of the Gaussian noise
    std: jnp.ndarray  # standard deviation of the Gaussian noise

    def __post_init__(self):
        assert self.mean.ndim == 1
        assert self.std.ndim == 1
        assert self.mean.shape == self.std.shape
        assert jnp.all(self.std >= 0)

    def sample(self, shape: Sequence[int] | int, key: jax.Array) -> jnp.ndarray:
        if isinstance(shape, int):
            shape = (shape,)

        return jax.random.normal(key, shape=shape) * self.std + self.mean

    def sub_gaussian_std(self) -> float:
        return self.std.max().item()


@dataclass(frozen=True)
class GaussianMixtureNoiseSpec(NoiseSpec):
    mean: Array  # mean of the Gaussian noise
    std: Array  # standard deviation of the Gaussian noise

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.mean.shape[1:]

    @property
    def n(self) -> int:
        assert len(self.mean) == len(self.std)
        return len(self.mean)

    def __post_init__(self):
        assert self.mean.shape == self.std.shape
        assert jnp.all(self.std >= 0)

    def sample(self, shape: Sequence[int] | int, key: Array) -> Array:
        if isinstance(shape, int):
            shape = (shape,)

        sample_shape = shape + (self.mean.shape[1:])

        key_1, key_2 = jax.random.split(key)

        components = jax.random.randint(key_1, shape=shape, minval=0, maxval=self.n)

        return jax.random.normal(key_2, shape=sample_shape) * self.std[components] + self.mean[components]

    def sub_gaussian_std(self) -> float:
        assert jnp.all(self.mean == 0)
        return self.std.max().item()
