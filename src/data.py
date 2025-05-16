from functools import partial
from typing import NamedTuple, Self, Iterator

import jax.numpy as jnp
from jax import Array
from jax.experimental.ode import odeint

from src.rkhs import RKHSFn


def collect_rotation(x_init: Array, n_rotations: float, n_samples: int, damping_factor: float) -> Array:
    assert x_init.ndim >= 1
    assert x_init.shape[-1] == 2

    omega = 2 * jnp.pi

    def rotation_ode(x: Array, _) -> Array:
        x1, x2 = x

        dx1 = -omega * x2 - damping_factor * x1
        dx2 = omega * x1 - damping_factor * x2

        return jnp.array([dx1, dx2])

    time = jnp.linspace(0, n_rotations, n_samples)

    @partial(jnp.vectorize, signature="(2)->(n,2)")
    def vectorized(x: Array) -> Array:
        return odeint(rotation_ode, x, time)

    return vectorized(x_init)


class RKHSFnSampling(NamedTuple):
    xs: Array
    ys: Array

    @classmethod
    def at_points(cls, fn: RKHSFn, xs: Array) -> Self:
        assert xs.ndim >= fn.kernel.ndim + 1
        ys = fn(xs)
        return cls(xs, ys)

    def add_y(self, delta_ys: Array) -> Self:
        return RKHSFnSampling(
            xs=self.xs,
            ys=self.ys + delta_ys
        )

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return RKHSFnSampling(self.xs[item], self.ys[item])

    def __iter__(self) -> Iterator[Self]:
        for i in range(len(self)):
            yield self[i]
