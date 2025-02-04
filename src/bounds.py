from functools import partial

import jax
import jax.numpy as jnp

from src.rkhs import Kernel, CKME


def compute_beta(
        rkhs_norm: float,
        sub_gaussian_std: float,
        regularization: float,
        kernel_matrix: jnp.ndarray,
        alpha: jnp.ndarray
) -> jnp.ndarray:
    n = kernel_matrix.shape[0]

    _, log_determinant = jnp.linalg.slogdet(jnp.eye(n) + kernel_matrix / regularization)

    return rkhs_norm + sub_gaussian_std / jnp.sqrt(regularization) * jnp.sqrt(
        log_determinant - 2 * jnp.log(alpha)
    )


def compute_sigma(kernel: Kernel, ckme: CKME, e: jnp.ndarray) -> jnp.ndarray:
    kernel_vector = kernel.many_one(ckme.xs, e)
    inv_kernel_vector = jax.scipy.linalg.cho_solve((ckme.cholesky, True), kernel_vector)

    return jnp.sqrt(kernel(e, e) - jnp.dot(kernel_vector, inv_kernel_vector))


def compute_sigmas(kernel: Kernel, ckme: CKME, es: jnp.ndarray) -> jnp.ndarray:
    @partial(jax.vmap)
    def batch_fn(e: jnp.ndarray) -> jnp.ndarray:
        return compute_sigma(kernel, ckme, e)

    return batch_fn(es)


@partial(jax.jit, static_argnums={0})
def analytical_cmmd_threshold(
        kernel: Kernel,
        ckme_1: CKME, ckme_2: CKME,
        e_1: jnp.ndarray, e_2: jnp.ndarray,
        rkhs_norm_1: float, rkhs_norm_2: float,
        sub_gaussian_std_1: float, sub_gaussian_std_2: float,
        power: float | jnp.ndarray
):
    kernel_matrix_1 = kernel.many_many(ckme_1.xs, ckme_1.xs)
    kernel_matrix_2 = kernel.many_many(ckme_2.xs, ckme_2.xs)

    sigma_1 = compute_sigma(kernel, ckme_1, e_1)
    sigma_2 = compute_sigma(kernel, ckme_2, e_2)

    @partial(jax.jit)
    def bound(alpha_1: jnp.ndarray, alpha_2: jnp.ndarray):
        beta_1 = compute_beta(rkhs_norm_1, sub_gaussian_std_1, ckme_1.regularization, kernel_matrix_1, alpha_1)
        beta_2 = compute_beta(rkhs_norm_2, sub_gaussian_std_2, ckme_2.regularization, kernel_matrix_2, alpha_2)

        return beta_1 * sigma_1 + beta_2 * sigma_2

    ts = jnp.linspace(0, 1, 100)[1:-1]
    bounds = bound(ts * power, (1 - ts) * power)
    return jnp.min(bounds)


def analytical_cmmd_thresholds(
        kernel: Kernel,
        ckme_1: CKME, ckme_2: CKME,
        es_1: jnp.ndarray, es_2: jnp.ndarray,
        rkhs_norm_1: float, rkhs_norm_2: float,
        sub_gaussian_std_1: float, sub_gaussian_std_2: float,
        power: float | jnp.ndarray
) -> jnp.ndarray:
    @partial(jax.vmap)
    def batch_fn(e_1: jnp.ndarray, e_2: jnp.ndarray) -> jnp.ndarray:
        return analytical_cmmd_threshold(
            kernel=kernel,
            ckme_1=ckme_1, ckme_2=ckme_2,
            e_1=e_1, e_2=e_2,
            rkhs_norm_1=rkhs_norm_1, rkhs_norm_2=rkhs_norm_2,
            sub_gaussian_std_1=sub_gaussian_std_1, sub_gaussian_std_2=sub_gaussian_std_2,
            power=power
        )

    return batch_fn(es_1, es_2)


def _bootstrap_datasets(
        xs: jnp.ndarray, ys: jnp.ndarray,
        n: int, size: int,
        key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    indices = jnp.arange(xs.shape[0])
    bootstrapped_indices = jax.random.choice(key, indices, shape=(n, size,), replace=True)

    return xs[bootstrapped_indices], ys[bootstrapped_indices]


def bootstrap_beta(
        kernel_x: Kernel,
        kernel_y: Kernel,
        xs_1: jnp.ndarray, ys_1: jnp.ndarray,
        xs_2: jnp.ndarray, ys_2: jnp.ndarray,
        dataset_size_1: int, dataset_size_2: int,
        n_bootstrap: int,
        regularization_1: float, regularization_2: float,
        alpha: jnp.ndarray | float,
        key: jax.Array
) -> jnp.ndarray:
    es = jnp.concatenate([xs_1, xs_2], axis=0)

    n_bootstrap_1 = n_bootstrap // 2
    n_bootstrap_2 = n_bootstrap - n_bootstrap_1

    def bootstrap_ckmes(
            size: int,
            regularization: float,
            key_: jax.Array
    ) -> CKME:
        key_1_, key_2_ = jax.random.split(key_, 2)
        bootstrapped_xs_1, bootstrapped_ys_1 = _bootstrap_datasets(xs_1, ys_1, n_bootstrap_1, size, key_1_)
        bootstrapped_xs_2, bootstrapped_ys_2 = _bootstrap_datasets(xs_2, ys_2, n_bootstrap_2, size, key_2_)

        bootstrapped_xs = jnp.concatenate([bootstrapped_xs_1, bootstrapped_xs_2], axis=0)
        bootstrapped_ys = jnp.concatenate([bootstrapped_ys_1, bootstrapped_ys_2], axis=0)

        return kernel_x.ckmes(bootstrapped_xs, bootstrapped_ys, regularization)

    key_1, key_2 = jax.random.split(key)

    ckmes_1 = bootstrap_ckmes(dataset_size_1, regularization_1, key_1)
    ckmes_2 = bootstrap_ckmes(dataset_size_2, regularization_2, key_2)

    @partial(jax.vmap)
    def batch_fn(x: jnp.ndarray) -> jnp.ndarray:
        kmes_1 = kernel_x.condition.many_one(ckmes_1, x)
        kmes_2 = kernel_x.condition.many_one(ckmes_2, x)

        cmmds = kernel_y.distance.batch(kmes_1, kmes_2)

        @partial(jax.vmap)
        def batch_sigma_fn(ckme: CKME) -> jnp.ndarray:
            return compute_sigma(kernel_x, ckme, x)

        sigmas_1 = batch_sigma_fn(ckmes_1)
        sigmas_2 = batch_sigma_fn(ckmes_2)
        sigmas = sigmas_1 + sigmas_2

        bootstrapped_betas = cmmds / sigmas

        return bootstrapped_betas

    betas_batch = batch_fn(es)

    return jnp.quantile(betas_batch.max(axis=0), 1 - alpha, axis=0)


def bootstrap_betas(
        kernel_x: Kernel, kernel_y: Kernel,
        xs_1: jnp.ndarray, ys_1: jnp.ndarray,
        xs_2: jnp.ndarray, ys_2: jnp.ndarray,
        regularization_1: float, regularization_2: float,
        n_bootstrap: int,
        alpha: jnp.ndarray | float,
        key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    es = jnp.concatenate([xs_1, xs_2], axis=0)

    def bootstrap_ckmes(
            xs: jnp.ndarray, ys: jnp.ndarray,
            size: int,
            regularization: float,
            key_: jax.Array
    ) -> CKME:
        bootstrapped_xs, bootstrapped_ys = _bootstrap_datasets(xs, ys, n_bootstrap, size, key_)
        return kernel_x.ckmes(bootstrapped_xs, bootstrapped_ys, regularization)

    dataset_size_1 = xs_1.shape[0]
    dataset_size_2 = xs_2.shape[0]

    def bootstrap(xs: jnp.ndarray, ys: jnp.ndarray, key_: jax.Array) -> jnp.ndarray:
        key_1_, key_2_ = jax.random.split(key_, 2)

        ckmes_1 = bootstrap_ckmes(xs, ys, dataset_size_1, regularization_1, key_1_)
        ckmes_2 = bootstrap_ckmes(xs, ys, dataset_size_2, regularization_2, key_2_)

        @partial(jax.vmap)
        def beta_batch_fn(x: jnp.ndarray) -> jnp.ndarray:
            kmes_1 = kernel_x.condition.many_one(ckmes_1, x)
            kmes_2 = kernel_x.condition.many_one(ckmes_2, x)

            cmmds = kernel_y.distance.batch(kmes_1, kmes_2)

            @partial(jax.vmap)
            def batch_sigma_fn(ckme: CKME) -> jnp.ndarray:
                return compute_sigma(kernel_x, ckme, x)

            sigmas_1 = batch_sigma_fn(ckmes_1)
            sigmas_2 = batch_sigma_fn(ckmes_2)
            sigmas = sigmas_1 + sigmas_2

            bootstrapped_betas = cmmds / sigmas

            return bootstrapped_betas

        return beta_batch_fn(es)

    key_1, key_2 = jax.random.split(key, 2)

    @partial(jax.vmap)
    def betas_batch_fn(alpha_1: jnp.ndarray, alpha_2: jnp.ndarray) -> jnp.ndarray:
        beta_1 = jnp.quantile(bootstrap(xs_1, ys_1, key_1).max(axis=0), 1 - alpha_1, axis=0)
        beta_2 = jnp.quantile(bootstrap(xs_2, ys_2, key_2).max(axis=0), 1 - alpha_2, axis=0)
        return jnp.array([beta_1, beta_2])

    ts = jnp.linspace(0, 1, 100)[1:-1]
    betas = betas_batch_fn(ts * alpha, (1 - ts) * alpha)

    norms = (betas ** 2).sum(axis=-1)
    best_betas = betas[jnp.argmin(norms)]

    return best_betas[0], best_betas[1]


@partial(jax.jit, static_argnums={0, 1, 2, 8})
def bootstrap_cmmd_threshold(
        kernel_x: Kernel, kernel_y: Kernel,
        n_bootstrap: int,
        ckme_1: CKME, ckme_2: CKME,
        e_1: jnp.ndarray, e_2: jnp.ndarray,
        power: float | jnp.ndarray,
        single_beta: bool,
        key: jax.Array
) -> jnp.ndarray:
    if single_beta:
        beta_1 = beta_2 = bootstrap_beta(
            kernel_x=kernel_x, kernel_y=kernel_y,
            xs_1=ckme_1.xs, ys_1=ckme_1.ys,
            xs_2=ckme_2.xs, ys_2=ckme_2.ys,
            dataset_size_1=ckme_1.xs.shape[0], dataset_size_2=ckme_2.ys.shape[0],
            n_bootstrap=n_bootstrap,
            regularization_1=ckme_1.regularization, regularization_2=ckme_2.regularization,
            alpha=power,
            key=key
        )
    else:
        beta_1, beta_2 = bootstrap_betas(
            kernel_x=kernel_x, kernel_y=kernel_y,
            xs_1=ckme_1.xs, ys_1=ckme_1.ys,
            xs_2=ckme_2.xs, ys_2=ckme_2.ys,
            regularization_1=ckme_1.regularization, regularization_2=ckme_2.regularization,
            n_bootstrap=n_bootstrap,
            alpha=power,
            key=key
        )

    sigma_1 = compute_sigma(kernel_x, ckme_1, e_1)
    sigma_2 = compute_sigma(kernel_x, ckme_2, e_2)

    return beta_1 * sigma_1 + beta_2 * sigma_2


def bootstrap_cmmd_thresholds(
        kernel_x: Kernel, kernel_y: Kernel,
        n_bootstrap: int,
        ckme_1: CKME, ckme_2: CKME,
        es_1: jnp.ndarray, es_2: jnp.ndarray,
        power: float | jnp.ndarray,
        single_beta: bool,
        key: jax.Array
) -> jnp.ndarray:
    @partial(jax.vmap)
    def batch_fn(e_1: jnp.ndarray, e_2: jnp.ndarray) -> jnp.ndarray:
        return bootstrap_cmmd_threshold(
            kernel_x=kernel_x, kernel_y=kernel_y,
            n_bootstrap=n_bootstrap,
            ckme_1=ckme_1, ckme_2=ckme_2,
            e_1=e_1, e_2=e_2,
            power=power,
            single_beta=single_beta,
            key=key
        )

    return batch_fn(es_1, es_2)
