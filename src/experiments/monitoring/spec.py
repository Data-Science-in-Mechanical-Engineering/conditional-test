from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm

from src.rkhs import VectorKernel, LinearKernel
from src.rkhs.testing import TestOutcome
from src.spec import GaussianNoiseSpec, BootstrapTestSpec, SpaceSpec


def sample_orthogonal_dynamics(dim: int, key: Array) -> Array:
    """ Samples a random orthogonal matrix uniformly from the orthogonal group O(dim) wrt the Haar measure.

    :param dim: The dimension of the orthogonal matrix.
    :param key: Random key.
    :return: An orthogonal matrix of shape (dim, dim).
    """
    matrix = jax.random.normal(key, shape=(dim, dim))
    q, r = jnp.linalg.qr(matrix)
    diagonal = jnp.diag(r)
    sign = jnp.sign(diagonal)
    return q * sign


def disturb_orthogonal_dynamics(matrix: Array, magnitude: float, key: Array) -> Array:
    """ Randomly disturbs an orthogonal matrix such that the resulting matrix is still orthogonal.

    :param matrix: An orthogonal matrix.
    :param magnitude: The geodesic distance between the original and the disturbed matrix.
    :param key: Random key.
    :return: The disturbed orthogonal matrix (orthogonal and of specified geodesic distance to the original matrix).
    """
    disturbance = jax.random.normal(key, matrix.shape)
    disturbance = (disturbance - disturbance.T) / 2
    disturbance = disturbance / jnp.linalg.norm(disturbance, ord="fro", axis=(-1, -2), keepdims=True) * magnitude
    return matrix @ expm(disturbance)


@dataclass(frozen=True)
class Config:
    regularization_reference: float
    regularization_windows: float
    dim: int
    disturbance: float
    n_repetitions: int
    n_reference_trajectories: int
    length_reference_trajectory: int
    length_nominal_trajectory: int
    length_anomalous_trajectory: int
    noise: GaussianNoiseSpec
    test: BootstrapTestSpec
    window_size: int
    seed: int

    @property
    def t_change(self):
        return self.length_nominal_trajectory - self.window_size

    @property
    def t_adapted(self):
        return self.length_nominal_trajectory

    @property
    def space(self) -> SpaceSpec:
        boundary = jnp.ones(self.dim) / jnp.sqrt(self.dim)
        return SpaceSpec(dim=self.dim, min=-boundary, max=boundary)

    @property
    def x_init(self) -> Array:
        return self.space.max_vector

    def __post_init__(self):
        assert self.dim >= 1
        assert self.n_repetitions >= 1
        assert self.length_reference_trajectory >= 1
        assert self.length_nominal_trajectory >= 1
        assert self.length_anomalous_trajectory >= 1
        assert self.window_size >= 1
        assert self.disturbance >= 0.0

    def vector_kernels(self) -> tuple[VectorKernel, VectorKernel]:
        kernel_y = LinearKernel(ndim=1)

        kernel_reference = VectorKernel(
            x=LinearKernel(ndim=1),
            y=kernel_y,
            regularization=self.regularization_reference
        )

        kernel_windows = VectorKernel(
            x=LinearKernel(ndim=1),
            y=kernel_y,
            regularization=self.regularization_windows
        )

        return kernel_reference, kernel_windows

    def sample_dynamics(self, key: Array) -> Array:
        return sample_orthogonal_dynamics(self.dim, key)

    def disturb_dynamics(self, dynamics: Array, key: Array) -> Array:
        return disturb_orthogonal_dynamics(dynamics, self.disturbance, key)

    def _sample_trajectory(self, dynamics: Array, x_init: Array, length: int, key: Array) -> Array:
        def step(state: Array, key_: Array) -> tuple[Array, Array]:
            next_state = jnp.einsum("ij,...j->...i", dynamics, state)
            next_state = next_state + self.noise.sample(next_state.shape, key_)
            return next_state, next_state

        keys = jax.random.split(key, length)
        _, trajectory = jax.lax.scan(f=step, init=x_init, xs=keys)
        trajectory = jnp.concatenate([x_init[None], trajectory], axis=0)

        return trajectory

    def sample_reference_dataset(self, dynamics: Array, key: Array) -> Array:
        xs_init = jnp.repeat(self.x_init[None], self.n_reference_trajectories, axis=0)
        trajectories = self._sample_trajectory(dynamics, xs_init, length=self.length_reference_trajectory, key=key)
        return trajectories.transpose(1, 0, 2).reshape(-1, self.dim)

    def sample_online_trajectory(self, dynamics: Array, dynamics_disturbed: Array, key: Array) -> Array:
        key_1, key_2 = jax.random.split(key)

        trajectory_nominal = self._sample_trajectory(
            dynamics=dynamics, x_init=self.x_init, length=self.length_nominal_trajectory, key=key_1
        )

        trajectory_anomalous = self._sample_trajectory(
            dynamics=dynamics_disturbed, x_init=trajectory_nominal[-1], length=self.length_anomalous_trajectory,
            key=key_2
        )

        return jnp.concatenate([trajectory_nominal, trajectory_anomalous[1:]], axis=0)

    def make_windows(self, trajectory: Array) -> tuple[Array, Array, Array]:
        assert trajectory.ndim == 2

        xs = trajectory[:-1]
        ys = trajectory[1:]

        length_trajectory = trajectory.shape[0] - 1
        window_indices = jnp.arange(self.window_size) + jnp.arange(length_trajectory - self.window_size + 1)[:, None]

        xs_windows = xs[window_indices]
        ys_windows = ys[window_indices]

        return xs_windows, ys_windows, xs_windows


@dataclass(frozen=True)
class SingleResult:
    outcomes: TestOutcome
    reference_dataset: Array
    online_trajectory: Array
    beta: Array
    posterior_std_reference: Array
    posterior_std_windows: Array


@dataclass(frozen=True)
class MultipleResult:
    runs: list[SingleResult]
    dynamics: Array
    dynamics_disturbed: Array

    def max_ratios(self) -> Array:
        return jnp.stack([
            run.outcomes.distance / run.outcomes.threshold
            for run in self.runs
        ]).max(axis=-1)

    def reference_mean_std(self) -> Array:
        return jnp.stack([
            run.posterior_std_reference
            for run in self.runs
        ]).mean(axis=-1)


DEFAULT_ARGS = dict(
    n_repetitions=100,
    regularization_reference=0.01,
    regularization_windows=0.01,
    n_reference_trajectories=5,
    length_reference_trajectory=400,
    length_nominal_trajectory=200,
    length_anomalous_trajectory=200,
    noise=GaussianNoiseSpec(mean=jnp.array([0]), std=jnp.array([0.01])),
    test=BootstrapTestSpec(
        n_bootstrap=100,
        significance_level=0.05
    ),
    window_size=50
)
