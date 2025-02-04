from __future__ import annotations

from jax import numpy as jnp
from typing import Literal

from src.rkhs.base import Kernel


class LinearKernel(Kernel):
    def __init__(self, ndim: int = Literal[0, 1]):
        assert ndim in {0, 1}

        def transform(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            return jnp.dot(x_1, x_2)

        super().__init__(fn=transform, ndim=ndim)


class PolynomialKernel(Kernel):
    def __init__(self, degree: int, ndim: int = 1):
        assert ndim in {0, 1}

        def transform(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            return (1 + jnp.dot(x_1, x_2)) ** degree

        super().__init__(fn=transform, ndim=ndim)


class GaussianKernel(Kernel):
    bandwidth: jnp.ndarray

    def __init__(self, bandwidth: float | jnp.ndarray, ndim: int = 1):
        bandwidth = jnp.array(bandwidth)
        assert bandwidth.ndim <= 1, "Bandwidth must be a scalar or a vector."

        self.bandwidth = bandwidth

        def transform(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            difference = (x_1 - x_2) / bandwidth
            return jnp.exp(-jnp.dot(difference, difference) / 2)

        super().__init__(fn=transform, ndim=ndim)
