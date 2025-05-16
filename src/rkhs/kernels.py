from __future__ import annotations

from functools import partial
from multiprocessing import Array
from typing import Self

from jax import numpy as jnp

from src.rkhs.base import Kernel


def _linear_kernel(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(x_1, x_2)


def _polynomial_kernel(x_1: jnp.ndarray, x_2: jnp.ndarray, degree: int) -> jnp.ndarray:
    return (1 + jnp.dot(x_1, x_2)) ** degree


def _gaussian_kernel(x_1: jnp.ndarray, x_2: jnp.ndarray, bandwidth: jnp.ndarray) -> jnp.ndarray:
    difference = (x_1 - x_2) / bandwidth
    return jnp.exp(-jnp.dot(difference, difference) / 2)


class LinearKernel(Kernel):
    def __init__(self, ndim: int = 1):
        if ndim not in {0, 1}:
            raise ValueError(f"Linear kernel only supports scalar or 1D inputs. Got {ndim}.")

        super().__init__(fn=_linear_kernel, ndim=ndim)


class PolynomialKernel(Kernel):
    degree: int

    def __init__(self, degree: int, ndim: int = 1):
        if ndim not in {0, 1}:
            raise ValueError(f"Polynomial kernel only supports scalar or 1D inputs. Got {ndim}.")

        super().__init__(fn=partial(_polynomial_kernel, degree=degree), ndim=ndim)
        self.degree = degree


class GaussianKernel(Kernel):
    bandwidth: jnp.ndarray

    def __init__(self, bandwidth: float | Array, ndim: int = 1):
        bandwidth = jnp.array(bandwidth)

        if bandwidth.ndim > 1:
            raise ValueError(f"Bandwidth must be a scalar or a vector. Got {bandwidth.ndim}.")
        if ndim not in {0, 1}:
            raise ValueError(f"Gaussian kernel only supports scalar or 1D inputs. Got {ndim}.")

        self.bandwidth = bandwidth

        super().__init__(fn=partial(_gaussian_kernel, bandwidth=bandwidth), ndim=ndim)
