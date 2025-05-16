from __future__ import annotations

from abc import ABC
from functools import partial
from typing import Callable, Final, NamedTuple, Self, Iterator, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from src.rkhs.util import _make_arg_signature

type KernelFn = Callable[[Array, Array], ArrayLike]


@partial(jax.tree_util.register_pytree_node_class)
class RKHSFn:
    kernel: Kernel
    coefficients: Array
    points: Array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coefficients.shape[:-1]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape_point(self) -> tuple[int, ...]:
        return self.points.shape[self.ndim + 1:]

    @property
    def n_points(self) -> int:
        return self.points.shape[self.ndim]

    def __init__(self, kernel: Kernel, coefficients: Array, points: Array):
        self.kernel = kernel
        self.coefficients = coefficients
        self.points = points

    def tree_flatten(self):
        children = (self.coefficients, self.points)
        aux_data = self.kernel
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Kernel, children: tuple[Array, Array]) -> Self:
        coefficients, points = children
        return cls(aux_data, coefficients, points)

    def reshape(self, *shape: int) -> Self:
        points = self.points.reshape(*shape, self.n_points, *self.shape_point)
        coefficients = self.coefficients.reshape(*shape, self.n_points)
        return RKHSFn(kernel=self.kernel, coefficients=coefficients, points=points)

    def transpose(self, *axes: int) -> Self:
        if set(axes) != set(range(self.ndim)):
            raise ValueError(f"Dimensions are not a permutation of {tuple(range(self.ndim))}. Got {axes}.")

        points = self.points.transpose(*axes, *(i for i in range(self.ndim, self.points.ndim)))
        coefficients = self.coefficients.transpose(*axes, self.ndim)
        return RKHSFn(kernel=self.kernel, coefficients=coefficients, points=points)

    @partial(jax.jit)
    def __call__(self, x: Array) -> Array:
        if x.ndim < self.ndim:
            raise ValueError(f"Cannot handle input of shape {x.shape}. Kernel has dimension {self.ndim}.")

        arg_signature_xs = _make_arg_signature(self.kernel.ndim, var_symbol="xs_", prefix='n')
        arg_signature_x = _make_arg_signature(self.kernel.ndim, var_symbol="x_")

        @partial(jnp.vectorize, signature=f"({arg_signature_xs}),(n),({arg_signature_x})->()")
        def vectorized(xs: Array, coefficients: Array, x_: Array) -> Array:
            kernel_vector = self.kernel(xs, x_)
            return jnp.dot(coefficients, kernel_vector)

        return vectorized(self.points, self.coefficients, x)

    @partial(jax.jit)
    def __add__(self, other: Self) -> Self:
        if not self.kernel == other.kernel:
            raise ValueError(f"Kernels must match. Got {self.kernel} and {other.kernel}.")

        points = jnp.concatenate([self.points, other.points], axis=self.ndim)
        coefficients = jnp.concatenate([self.coefficients, other.coefficients], axis=self.ndim)

        return RKHSFn(kernel=self.kernel, coefficients=coefficients, points=points)

    @partial(jax.jit)
    def __mul__(self, scalars: ArrayLike) -> Self:
        scalars = jnp.asarray(scalars)

        coefficients = scalars * self.coefficients
        points = jnp.broadcast_to(self.points, coefficients.shape[:-1] + (self.n_points,) + self.shape_point)

        return RKHSFn(kernel=self.kernel, coefficients=coefficients, points=points)

    @partial(jax.jit)
    def __rmul__(self, scalar: ArrayLike) -> Self:
        return self.__mul__(scalar)

    def __getitem__(self, item) -> Self:
        indexing = jnp.indices(self.shape)
        indexing = tuple(index[item] for index in indexing)

        points = self.points[*indexing]
        coefficients = self.coefficients[*indexing]
        return RKHSFn(kernel=self.kernel, coefficients=coefficients, points=points)

    def __iter__(self) -> Iterator[Self]:
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        if self.ndim == 0:
            return 0

        return int(self.shape[0])


@partial(jax.tree_util.register_pytree_node_class)
class CME:
    kernel: VectorKernel
    xs: Array
    ys: Array
    gram: Array
    cholesky: Array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.xs.shape[:-self.kernel.x.ndim - 1]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape_point_x(self) -> tuple[int, ...]:
        return self.xs.shape[self.ndim + 1:]

    @property
    def shape_point_y(self) -> tuple[int, ...]:
        return self.ys.shape[self.ndim + 1:]

    @property
    def n_points(self) -> int:
        return self.xs.shape[self.ndim]

    def __init__(self, kernel: VectorKernel, xs: Array, ys: Array, gram: Array, cholesky: Array):
        self.kernel = kernel
        self.xs = xs
        self.ys = ys
        self.gram = gram
        self.cholesky = cholesky

    def tree_flatten(self):
        children = self.xs, self.ys, self.gram, self.cholesky
        aux_data = self.kernel
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: VectorKernel, children: tuple[Array, Array, Array, Array]) -> Self:
        xs, ys, gram, cholesky = children
        return cls(aux_data, xs, ys, gram, cholesky)

    def reshape(self, *shape: int) -> Self:
        xs = self.xs.reshape(*shape, self.n_points, *self.shape_point_x)
        ys = self.ys.reshape(*shape, self.n_points, *self.shape_point_y)
        gram = self.gram.reshape(*shape, self.n_points, self.n_points)
        cholesky = self.cholesky.reshape(*shape, self.n_points, self.n_points)

        return CME(self.kernel, xs, ys, gram, cholesky)

    def transpose(self, *axes: int) -> Self:
        if set(axes) != set(range(self.ndim)):
            raise ValueError(f"Dimensions are not a permutation of {tuple(range(self.ndim))}. Got {axes}.")

        xs = self.xs.transpose(*axes, *(i for i in range(self.ndim, self.xs.ndim)))
        ys = self.ys.transpose(*axes, *(i for i in range(self.ndim, self.ys.ndim)))
        gram = self.gram.transpose(*axes, self.ndim, self.ndim + 1)
        cholesky = self.cholesky.transpose(*axes, self.ndim, self.ndim + 1)

        return CME(self.kernel, xs, ys, gram, cholesky)

    def influence(self, x: Array) -> Array:
        arg_signature_xs = _make_arg_signature(self.kernel.x.ndim, "xs_", prefix='n')
        arg_signature_x = _make_arg_signature(self.kernel.x.ndim, "x_")

        @partial(jnp.vectorize, signature=f"({arg_signature_xs}),(n,n),({arg_signature_x})->(n)")
        def vectorized_coefficients(xs: Array, cholesky: Array, x_: Array):
            kernel_vector = self.kernel.x(xs, x_)
            return jax.scipy.linalg.cho_solve((cholesky, True), kernel_vector)

        return vectorized_coefficients(self.xs, self.cholesky, x)

    @partial(jax.jit)
    def __call__(self, x: Array) -> RKHSFn:
        if x.ndim < self.kernel.x.ndim:
            raise ValueError(f"Cannot handle input of shape {x.shape}. Kernel has dimension {self.kernel.x.ndim}.")

        coefficients = self.influence(x)
        ys = jnp.broadcast_to(self.ys, coefficients.shape[:-1] + (self.n_points,) + self.shape_point_y)

        return RKHSFn(kernel=self.kernel.y, coefficients=coefficients, points=ys)

    def __getitem__(self, item) -> Self:
        indexing = jnp.indices(self.shape)
        indexing = tuple(index[item] for index in indexing)

        xs = self.xs[*indexing]
        ys = self.ys[*indexing]
        gram = self.gram[*indexing]
        cholesky = self.cholesky[*indexing]
        return CME(self.kernel, xs, ys, gram, cholesky)

    def __iter__(self) -> Iterator[Self]:
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        if self.ndim == 0:
            return 0

        return int(self.shape[0])


class Kernel(ABC):
    __fn: Final[KernelFn]
    ndim: Final[int]

    def __init__(self, fn: KernelFn, ndim: int):
        self.__fn = fn
        self.ndim = ndim

    def function(self, coefficients: Array, points: Array) -> RKHSFn:
        if points.ndim < self.ndim + 1:
            raise ValueError(f"Points must have at least {self.ndim + 1} dimensions. Got {points.ndim}.")

        if coefficients.ndim != points.ndim - 1:
            raise ValueError(f"Inconsistent dimensions. Coefficients have {coefficients.ndim} dimensions, but points "
                             f"have {points.ndim} dimensions.")

        return RKHSFn(self, coefficients, points)

    @partial(jax.jit, static_argnums={0})
    def kme(self, xs: Array) -> RKHSFn:
        n = xs.shape[xs.ndim - self.ndim - 1]
        coefficients = jnp.full(shape=xs.shape[:xs.ndim - self.ndim], fill_value=1.0 / n)
        return self.function(coefficients, xs)

    @partial(jax.jit, static_argnums={0})
    def kernel_matrix(self, xs_1: Array, xs_2: Array) -> Array:
        if xs_1.ndim < self.ndim + 1:
            raise ValueError(f"Dataset for kernel of dimension {self.ndim} must have at least dimension "
                             f"{self.ndim + 1}.")
        if xs_2.ndim < self.ndim + 1:
            raise ValueError(f"Dataset for kernel of dimension {self.ndim} must have at least dimension "
                             f"{self.ndim + 1}.")

        arg_signature_1 = _make_arg_signature(self.ndim, "x1_", 'n')
        arg_signature_2 = _make_arg_signature(self.ndim, "x2_", 'm')

        @partial(jnp.vectorize, signature=f"({arg_signature_1}),({arg_signature_2})->(n,m)")
        def vectorized(xs_1_: Array, xs_2_: Array) -> Array:
            return self(xs_1_[:, None], xs_2_[None, :])

        return vectorized(xs_1, xs_2)

    def gram(self, xs: Array) -> Array:
        return self.kernel_matrix(xs, xs)

    @partial(jax.jit, static_argnums={0})
    def __call__(self, x_1: Array, x_2: Array) -> ArrayLike:
        if x_1.ndim < self.ndim:
            raise ValueError(f"Cannot handle input 1 of shape {x_1.shape}. Kernel has dimension {self.ndim}.")
        if x_2.ndim < self.ndim:
            raise ValueError(f"Cannot handle input 2 of shape {x_2.shape}. Kernel has dimension {self.ndim}.")

        arg_signature_1 = _make_arg_signature(self.ndim, 'x1_')
        arg_signature_2 = _make_arg_signature(self.ndim, 'x2_')

        @partial(jnp.vectorize, signature=f"({arg_signature_1}),({arg_signature_2})->()")
        def vectorized(x_1_: Array, x_2_: Array) -> ArrayLike:
            return self.__fn(x_1_, x_2_)

        return vectorized(x_1, x_2)


class VectorKernel(NamedTuple):
    x: Kernel
    y: Kernel
    regularization: float | Array

    @partial(jax.jit, static_argnums={0})
    def cme(self, xs: Array, ys: Array, gram: Optional[Array] = None) -> CME:
        if xs.ndim < self.x.ndim + 1:
            raise ValueError(f"Cannot handle input 1 of shape {xs.shape}. Kernel has dimension {self.x.ndim}.")
        if ys.shape[:ys.ndim - self.y.ndim] != xs.shape[:xs.ndim - self.x.ndim]:
            raise ValueError(f"Inconsistent dimensions. Input for x has shape {xs.shape}, but input for y has shape "
                             f"{ys.shape}.")

        if gram is None:
            gram = self.x.gram(xs)

        diagonal_indices = jnp.arange(xs.shape[xs.ndim - self.x.ndim - 1])

        regularized_gram = gram.at[..., diagonal_indices, diagonal_indices].add(self.regularization)
        cholesky, _ = jax.scipy.linalg.cho_factor(regularized_gram, lower=True)

        return CME(self, xs, ys, gram, cholesky)


@partial(jax.jit)
def dot(fn_1: RKHSFn, fn_2: RKHSFn, kernel_matrix: Optional[Array] = None) -> Array:
    if fn_1.kernel != fn_2.kernel:
        raise ValueError(f"Kernels must match. Got {fn_1.kernel} and {fn_2.kernel}.")

    if kernel_matrix is None:
        kernel_matrix = fn_1.kernel.kernel_matrix(fn_1.points, fn_2.points)

    return jnp.einsum("...i,...ij,...j->...", fn_1.coefficients, kernel_matrix, fn_2.coefficients)


@partial(jax.jit)
def squared_distance(
        fn_1: RKHSFn, fn_2: RKHSFn, kernel_matrix_11: Optional[Array] = None, kernel_matrix_22: Optional[Array] = None,
        kernel_matrix_12: Optional[Array] = None
) -> Array:
    dp_11 = dot(fn_1, fn_1, kernel_matrix=kernel_matrix_11)
    dp_22 = dot(fn_2, fn_2, kernel_matrix=kernel_matrix_22)
    dp_12 = dot(fn_1, fn_2, kernel_matrix=kernel_matrix_12)

    return dp_11 + dp_22 - 2 * dp_12


def distance(
        fn_1: RKHSFn, fn_2: RKHSFn, kernel_matrix_11: Optional[Array] = None, kernel_matrix_22: Optional[Array] = None,
        kernel_matrix_12: Optional[Array] = None
) -> Array:
    squared_distance_ = squared_distance(
        fn_1, fn_2,
        kernel_matrix_11=kernel_matrix_11, kernel_matrix_22=kernel_matrix_22, kernel_matrix_12=kernel_matrix_12
    )

    return jnp.sqrt(jnp.clip(squared_distance_, min=0))  # clip to avoid numerical errors


def squared_norm(fn: RKHSFn, kernel_matrix: Optional[Array] = None) -> Array:
    return dot(fn, fn, kernel_matrix=kernel_matrix)


def norm(fn: RKHSFn, kernel_matrix: Optional[Array] = None) -> Array:
    squared_norm_ = squared_norm(fn, kernel_matrix=kernel_matrix)
    return jnp.sqrt(jnp.clip(squared_norm_, min=0))  # clip to avoid numerical errors


@partial(jax.jit, static_argnums={0, 1})
def _wrap_rkhs_operator(
        operator: Callable[[RKHSFn, RKHSFn], ArrayLike], kernel: Kernel, xs_1: Array, xs_2: Array
) -> Array:
    fn_1 = kernel.kme(xs_1)
    fn_2 = kernel.kme(xs_2)
    return operator(fn_1, fn_2)


def kme_dot(kernel: Kernel, xs_1: Array, xs_2: Array) -> Array:
    return _wrap_rkhs_operator(dot, kernel, xs_1, xs_2)


def squared_mmd(kernel: Kernel, xs_1: Array, xs_2: Array) -> Array:
    return _wrap_rkhs_operator(squared_distance, kernel, xs_1, xs_2)


def mmd(kernel: Kernel, xs_1: Array, xs_2: Array) -> Array:
    return _wrap_rkhs_operator(distance, kernel, xs_1, xs_2)


@partial(jax.jit, static_argnums={0, 1})
def _wrap_conditional_rkhs_operator(
        operator: Callable[[RKHSFn, RKHSFn], ArrayLike], kernel: VectorKernel,
        xs_1: Array, xs_2: Array, ys_1: Array, ys_2: Array, e_1: Array, e_2: Array
) -> Array:
    cme_1 = kernel.cme(xs_1, ys_1)
    cme_2 = kernel.cme(xs_2, ys_2)
    kme_1 = cme_1(e_1)
    kme_2 = cme_2(e_2)
    return operator(kme_1, kme_2)


def cme_dot(kernel: VectorKernel, xs_1: Array, xs_2: Array, ys_1: Array, ys_2: Array, e_1: Array, e_2: Array) -> Array:
    return _wrap_conditional_rkhs_operator(dot, kernel, xs_1, xs_2, ys_1, ys_2, e_1, e_2)


def squared_cmmd(
        kernel: VectorKernel, xs_1: Array, xs_2: Array, ys_1: Array, ys_2: Array, e_1: Array, e_2: Array
) -> Array:
    return _wrap_conditional_rkhs_operator(squared_distance, kernel, xs_1, xs_2, ys_1, ys_2, e_1, e_2)


def cmmd(kernel: VectorKernel, xs_1: Array, xs_2: Array, ys_1: Array, ys_2: Array, e_1: Array, e_2: Array) -> Array:
    return _wrap_conditional_rkhs_operator(distance, kernel, xs_1, xs_2, ys_1, ys_2, e_1, e_2)
