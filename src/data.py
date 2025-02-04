from functools import partial
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from src.rkhs import Kernel, RKHSFn


class RKHSFnSampling(NamedTuple):
    xs: jnp.ndarray
    ys: jnp.ndarray

    @classmethod
    @partial(jax.jit, static_argnums={0, 2})
    def at_points(cls, fn: RKHSFn, kernel: Kernel, xs: jnp.ndarray) -> Self:
        ys = kernel.evaluate.one_many(fn, xs)
        return cls(xs, ys)

    @classmethod
    @partial(jax.jit, static_argnums={0, 2})
    def at_points_batch(cls, fn: RKHSFn, kernel: Kernel, xs_batch: jnp.ndarray) -> Self:
        @partial(jax.vmap)
        def batch_fn(xs: jnp.ndarray) -> Self:
            return cls.at_points(fn, kernel, xs)

        return batch_fn(xs_batch)

    def apply_gaussian_noise(self, std: float, key: jax.Array) -> Self:
        noise = jax.random.normal(key, shape=self.ys.shape) * std
        return self._replace(ys=self.ys + noise)

    def __getitem__(self, item) -> Self:
        return RKHSFnSampling(self.xs[item], self.ys[item])


def collect_rotation_trajectories(
        xs_init: jnp.ndarray,
        n_rotations: float,
        n_samples: int,
        damping_factor: float = 0.0,
) -> jnp.ndarray:
    assert xs_init.ndim == 2
    assert xs_init.shape[1] == 2

    omega = 2 * jnp.pi / 1

    def rotation_ode(x: jnp.ndarray, _):
        x1, x2 = x

        dx1 = -omega * x2 - damping_factor * x1
        dx2 = omega * x1 - damping_factor * x2

        return jnp.array([dx1, dx2])

    @partial(jax.vmap)
    def batch_fn(x_init: jnp.ndarray) -> jnp.ndarray:
        return odeint(rotation_ode, x_init, jnp.linspace(0, n_rotations, n_samples))

    return batch_fn(xs_init)


def collect_gradient_trajectories(
        kernel: Kernel,
        fn: RKHSFn,
        xs_init: jnp.ndarray,
        n_samples: int,
        t_start: float = 0.0,
        t_end: float = 1.0
) -> jnp.ndarray:
    assert xs_init.ndim == kernel.ndim + 1
    assert n_samples >= 1

    @jax.grad
    def gradient_fn(x: jnp.ndarray):
        return kernel.evaluate(fn, x)

    def gradient_ode(x: jnp.ndarray, _):
        return -gradient_fn(x)

    @partial(jax.vmap)
    def batch_fn(x_init: jnp.ndarray) -> jnp.ndarray:
        return odeint(gradient_ode, x_init, jnp.linspace(t_start, t_end, n_samples))

    return batch_fn(xs_init)
