from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from src.rkhs.base import RKHSFn, Kernel


@partial(jax.jit, static_argnums={2, 3, 4})
def uniform_rkhs_subspace_ball(
        key: Array, basis_points: Array, kernel: Kernel, radius: float, surface: bool = False
) -> RKHSFn:
    if basis_points.ndim < kernel.ndim + 1:
        raise ValueError("Basis points must have at least one more dimension than the kernel.")

    key_coefficients, key_norm = jax.random.split(key)
    coefficient_shape = basis_points.shape[:-kernel.ndim]
    norm_shape = coefficient_shape[:-1]

    gram_matrix = kernel.gram(basis_points)

    u, s, _ = jnp.linalg.svd(gram_matrix, hermitian=True)
    mask = s > 1e-5
    dimension = mask.sum(axis=-1)

    random_direction_orthonormal = jax.random.normal(key_coefficients, coefficient_shape) * mask

    actual_norm = jnp.linalg.norm(random_direction_orthonormal, axis=-1, keepdims=True)

    if surface:
        norm = jnp.full(norm_shape, radius)
    else:
        norm = radius * jax.random.uniform(key_norm, norm_shape, minval=0, maxval=1) ** (1 / dimension)

    coefficients_orthonormal_basis = random_direction_orthonormal / actual_norm * norm[..., None]
    coefficients = jnp.einsum("...ij,...j->...i", u, coefficients_orthonormal_basis / jnp.sqrt(s) * mask)

    return kernel.function(coefficients, basis_points)
