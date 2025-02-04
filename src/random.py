from functools import partial
from typing import Generator

import jax
import jax.numpy as jnp

from src.rkhs import Kernel, RKHSFn

jax.config.update("jax_enable_x64", True)


def generate_random_keys(seed: int) -> Generator[jax.Array, None, None]:
    key = jax.random.PRNGKey(seed)

    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def _uniform_point_basis_coefficients(
        gram_matrix: jnp.ndarray,
        radius: float,
        key: jax.Array,
        sphere: bool = False
) -> jnp.ndarray:
    assert gram_matrix.ndim == 2
    assert gram_matrix.shape[0] == gram_matrix.shape[1]

    key_coefficients, key_norm = jax.random.split(key)
    n_points = gram_matrix.shape[0]

    u, s, _ = jnp.linalg.svd(gram_matrix, hermitian=True)
    mask = s > 1e-8
    dimension = mask.sum()

    random_direction_orthonormal = jax.random.normal(key_coefficients, (n_points,)) * mask
    actual_norm = jnp.linalg.norm(random_direction_orthonormal)

    if sphere:
        norm = radius
    else:
        norm = radius * jax.random.uniform(key_norm, minval=0, maxval=1) ** (1 / dimension)

    coefficients_orthonormal_basis = random_direction_orthonormal / actual_norm * norm

    return u @ (coefficients_orthonormal_basis / jnp.sqrt(s) * mask)


@partial(jax.jit, static_argnums={0, 4})
def uniform_rkhs_subspace_fn(kernel: Kernel, xs: jnp.ndarray, radius: float, key: jax.Array, sphere: bool=False) -> RKHSFn:
    assert kernel.ndim == xs.ndim - 1

    gram_matrix = kernel.many_many(xs, xs)
    coefficients = _uniform_point_basis_coefficients(gram_matrix, radius, key, sphere)

    return RKHSFn(coefficients=coefficients, points=xs)


@partial(jax.jit, static_argnums={0, 4})
def uniform_rkhs_subspace_fns(kernel: Kernel, xs_batch: jnp.ndarray, radius: float, key: jax.Array, sphere: bool = False) -> RKHSFn:
    assert kernel.ndim == xs_batch.ndim - 2

    @partial(jax.vmap)
    def batch_fn(xs: jnp.ndarray, key_: jax.Array):
        return uniform_rkhs_subspace_fn(kernel, xs, radius, key_, sphere)

    return batch_fn(xs_batch, jax.random.split(key, xs_batch.shape[0]))
