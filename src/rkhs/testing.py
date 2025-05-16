from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Self, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from src.rkhs import CME, RKHSFn, VectorKernel, distance
from src.rkhs.util import _make_arg_signature


class TestOutcome(NamedTuple):
    distance: Array
    threshold: Array

    def rejection(self) -> Array:
        return self.distance > self.threshold


@partial(jax.jit)
def _compute_posterior_std(cme: CME, x: Array, influence_vector: Array) -> Array:
    arg_signature_xs = _make_arg_signature(cme.kernel.x.ndim, 'x', prefix='n')
    arg_signature_x = _make_arg_signature(cme.kernel.x.ndim, 'x')

    @partial(jnp.vectorize, signature=f"({arg_signature_xs}),({arg_signature_x}),(n)->()")
    def vectorized(xs_: Array, x_: Array, influence_vector_: Array) -> Array:
        kernel_vector = cme.kernel.x(xs_, x_)
        return jnp.sqrt(cme.kernel.x(x_, x_) - jnp.dot(kernel_vector, influence_vector_))

    return vectorized(cme.xs, x, influence_vector)


@partial(jax.jit)
def _extract_submatrix(matrix: Array, indices_1: Array, indices_2: Array) -> Array:
    return matrix[indices_2[..., None, :], indices_1[..., None]]


@partial(jax.jit, static_argnums={1})
def _bootstrap_cme(cme: CME, n_bootstrap: int, key: Array) -> tuple[CME, Array]:
    if cme.ndim != 0:
        raise ValueError("CME must be 0-dimensional for bootstrapping.")

    bootstrapped_indices = jax.random.randint(key, shape=(n_bootstrap, cme.n_points), minval=0, maxval=cme.n_points)

    def bootstrap(indices: Array) -> CME:
        return cme.kernel.cme(
            cme.xs[indices],
            cme.ys[indices],
            gram=_extract_submatrix(cme.gram, indices, indices),
        )

    bootstrapped_cmes = jax.lax.map(bootstrap, bootstrapped_indices)

    return bootstrapped_cmes, bootstrapped_indices


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class TestEmbedding(ABC):
    kme: RKHSFn

    @abstractmethod
    def _threshold(self, significance_level: float) -> Array:
        raise NotImplementedError

    def threshold(self, significance_level: ArrayLike) -> Array:
        if not (jnp.isscalar(significance_level) or significance_level.ndim == 1):
            raise ValueError(f"Significance level must be a scalar or a 1D array. Got shape: "
                             f"{significance_level.shape}")

        if jnp.isscalar(significance_level):
            return self._threshold(significance_level)

        @partial(jax.vmap)
        def batched(significance_level_: ArrayLike) -> Array:
            return self._threshold(significance_level_)

        return batched(significance_level)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class ConditionedTestEmbedding(TestEmbedding, ABC):
    posterior_std: Array

    @abstractmethod
    def beta(self, significance_level: float) -> Array:
        raise NotImplementedError

    def _threshold(self, significance_level: float) -> Array:
        beta = self.beta(significance_level)
        return beta * self.posterior_std


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class AnalyticalConditionedTestEmbedding(ConditionedTestEmbedding):
    regularization: Array
    log_determinant: Array
    rkhs_norm: Array
    sub_gaussian_std: Array

    def beta(self, significance_level: float) -> Array:
        return self.rkhs_norm + self.sub_gaussian_std / jnp.sqrt(self.regularization) * jnp.sqrt(
            self.log_determinant - 2 * jnp.log(significance_level)
        )


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class BootstrapConditionedTestEmbedding(ConditionedTestEmbedding):
    beta_null: Array

    def beta(self, significance_level: float) -> Array:
        return jnp.quantile(self.beta_null, 1 - significance_level, axis=-1)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class ConditionalTestEmbedding[T: ConditionedTestEmbedding](ABC):
    cme: CME

    def posterior_std(self, x: Array, influence_vector: Array | None = None) -> Array:
        if influence_vector is None:
            influence_vector = self.cme.influence(x)
        return _compute_posterior_std(self.cme, x, influence_vector)

    @abstractmethod
    def __call__(self, x: Array) -> T:
        raise NotImplementedError


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class AnalyticalConditionalTestEmbedding(ConditionalTestEmbedding[AnalyticalConditionedTestEmbedding]):
    log_determinant: Array
    rkhs_norm: Array
    sub_gaussian_std: Array

    @classmethod
    def from_data(
            cls, kernel: VectorKernel, xs: Array, ys: Array, rkhs_norm: ArrayLike, sub_gaussian_std: ArrayLike
    ) -> Self:
        cme = kernel.cme(xs, ys)
        return cls.from_cme(cme, rkhs_norm, sub_gaussian_std)

    @classmethod
    @partial(jax.jit, static_argnums={0})
    def from_cme(cls, cme: CME, rkhs_norm: ArrayLike, sub_gaussian_std: ArrayLike) -> Self:
        _, log_determinant = jnp.linalg.slogdet(jnp.eye(cme.n_points) + cme.gram / cme.kernel.regularization)

        rkhs_norm = jnp.broadcast_to(rkhs_norm, log_determinant.shape)
        sub_gaussian_std = jnp.broadcast_to(sub_gaussian_std, log_determinant.shape)

        return AnalyticalConditionalTestEmbedding(
            cme=cme, log_determinant=log_determinant, rkhs_norm=rkhs_norm,
            sub_gaussian_std=sub_gaussian_std
        )

    @partial(jax.jit)
    def __call__(self, x: Array) -> AnalyticalConditionedTestEmbedding:
        kme = self.cme(x)
        posterior_std = self.posterior_std(x, kme.coefficients)

        log_determinant = jnp.broadcast_to(self.log_determinant, kme.shape)
        regularization = jnp.broadcast_to(self.cme.kernel.regularization, kme.shape)
        rkhs_norm = jnp.broadcast_to(self.rkhs_norm, kme.shape)
        sub_gaussian_std = jnp.broadcast_to(self.sub_gaussian_std, kme.shape)

        return AnalyticalConditionedTestEmbedding(
            kme=kme, posterior_std=posterior_std, log_determinant=log_determinant, regularization=regularization,
            rkhs_norm=rkhs_norm, sub_gaussian_std=sub_gaussian_std
        )


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class BootstrapConditionalTestEmbedding(ConditionalTestEmbedding[BootstrapConditionedTestEmbedding]):
    beta_null: Array

    @classmethod
    def from_data(cls, kernel: VectorKernel, xs: Array, ys: Array, es: Array, n_bootstrap: int, key: Array) -> Self:
        cme = kernel.cme(xs, ys)
        return cls.from_cme(cme, es, n_bootstrap, key)

    @classmethod
    @partial(jax.jit, static_argnums={0, 3})
    def from_cme(cls, cme: CME, es: Array, n_bootstrap: int, key: Array) -> Self:
        def bootstrap_beta(cme_: CME, key_: Array) -> Array:
            key_1, key_2 = jax.random.split(key_)

            bootstrap_cme_1, bootstrap_indices_1 = _bootstrap_cme(cme_, 1, key_1)
            bootstrap_cme_2, bootstrap_indices_2 = _bootstrap_cme(cme_, 1, key_2)

            bootstrap_cme_1 = bootstrap_cme_1[0]
            bootstrap_cme_2 = bootstrap_cme_2[0]

            kernel_matrix_y = cme.kernel.y.kernel_matrix(cme_.ys, cme_.ys)

            # for each covariate: the mean embedding obtained by conditioning the CME on that point
            # shape: (dataset_size,)
            bootstrap_kmes_1 = bootstrap_cme_1(es)
            bootstrap_kmes_2 = bootstrap_cme_2(es)

            # for each covariate: the CMMD between the datasets at that point
            # shape: (dataset_size,)
            cmmds = distance(
                fn_1=bootstrap_kmes_1, fn_2=bootstrap_kmes_2,
                kernel_matrix_11=_extract_submatrix(kernel_matrix_y, bootstrap_indices_1, bootstrap_indices_1),
                kernel_matrix_22=_extract_submatrix(kernel_matrix_y, bootstrap_indices_2, bootstrap_indices_2),
                kernel_matrix_12=_extract_submatrix(kernel_matrix_y, bootstrap_indices_1, bootstrap_indices_2),
            )

            # for each covariate: the value of sigma for the CME conditioned on that point
            # shape: (n_bootstrap, dataset_size)
            bootstrap_posterior_std_1 = _compute_posterior_std(bootstrap_cme_1, es, bootstrap_kmes_1.coefficients)
            bootstrap_posterior_std_2 = _compute_posterior_std(bootstrap_cme_2, es, bootstrap_kmes_2.coefficients)

            # for each covariate: the smallest value of beta such that H_0 is not rejected
            # shape: (dataset_size,)
            beta = cmmds / (bootstrap_posterior_std_1 + bootstrap_posterior_std_2)

            # the smallest value of beta such that H_0 is not rejected at any point
            # shape: ()
            return beta.max()

        def bootstrap_betas(cme_: CME, key_: Array) -> Array:
            keys_ = jax.random.split(key_, n_bootstrap)
            return jax.lax.map(lambda key__: bootstrap_beta(cme_, key__), xs=keys_)

        cme_shape = cme.shape
        cmes = cme.reshape(-1)

        keys = jax.random.split(key, num=cmes.shape[0])
        beta_null = jax.lax.map(lambda inp: bootstrap_betas(*inp), (cmes, keys))

        beta_null = beta_null.reshape(*cme_shape, n_bootstrap)

        return BootstrapConditionalTestEmbedding(cme=cme, beta_null=beta_null)

    def __call__(self, x: Array) -> BootstrapConditionedTestEmbedding:
        kme = self.cme(x)
        posterior_std = self.posterior_std(x, kme.coefficients)
        return BootstrapConditionedTestEmbedding(kme=kme, posterior_std=posterior_std, beta_null=self.beta_null)


def two_sample_test(
        embedding_1: TestEmbedding, embedding_2: TestEmbedding, significance_level: ArrayLike
) -> TestOutcome:
    distance_ = distance(embedding_1.kme, embedding_2.kme)
    threshold = embedding_1.threshold(significance_level / 2) + embedding_2.threshold(significance_level / 2)

    return TestOutcome(distance=distance_, threshold=threshold)
