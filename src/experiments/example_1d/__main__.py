import jax
import jax.numpy as jnp

from src import expyro
from src.bounds import bootstrap_betas, compute_sigmas, compute_beta
from src.experiments.example_1d.plots import plot_analytical, plot_bootstrap
from src.experiments.example_1d.spec import Config, DEFAULT_CONFIG, Result
from src.figures.util import set_plot_style
from src.random import generate_random_keys
from src.util import DIR_RESULTS


@expyro.plot(plot_analytical, file_format="png")
@expyro.plot(plot_bootstrap, file_format="png")
@expyro.experiment(DIR_RESULTS, "example_1d")
def main(config: Config):
    rng = generate_random_keys(config.seed)

    kernel = config.kernel_parametrization()
    state_space = config.state_space()

    function_1 = config.sample_rkhs_fn(kernel.x, next(rng))
    function_2 = config.sample_rkhs_fn(kernel.x, next(rng))

    dataset_1 = config.sample_dataset(kernel.x, function_1, next(rng))
    dataset_2 = config.sample_dataset(kernel.x, function_2, next(rng))

    ckme_1 = kernel.x.ckme(dataset_1.xs, dataset_1.ys, kernel.regularization)
    ckme_2 = kernel.x.ckme(dataset_2.xs, dataset_2.ys, kernel.regularization)

    kmes_1 = kernel.x.condition.one_many(ckme_1, state_space)
    kmes_2 = kernel.x.condition.one_many(ckme_2, state_space)

    beta_1_bootstrap, beta_2_bootstrap = bootstrap_betas(
        kernel_x=kernel.x, kernel_y=kernel.y,
        xs_1=ckme_1.xs, ys_1=ckme_1.ys,
        xs_2=ckme_2.xs, ys_2=ckme_2.ys,
        regularization_1=ckme_1.regularization, regularization_2=ckme_2.regularization,
        n_bootstrap=config.n_bootstrap,
        alpha=config.confidence_level,
        key=next(rng)
    )

    beta_1_analytical = compute_beta(
        rkhs_norm=config.rkhs_ball_radius,
        sub_gaussian_std=config.noise.sub_gaussian_std(),
        regularization=ckme_1.regularization,
        kernel_matrix=kernel.x.many_many(ckme_1.xs, ckme_1.xs),
        alpha=jnp.array(config.confidence_level),
    )

    beta_2_analytical = compute_beta(
        rkhs_norm=config.rkhs_ball_radius,
        sub_gaussian_std=config.noise.sub_gaussian_std(),
        regularization=ckme_2.regularization,
        kernel_matrix=kernel.x.many_many(ckme_2.xs, ckme_2.xs),
        alpha=jnp.array(config.confidence_level),
    )

    sigma_1 = compute_sigmas(kernel=kernel.x, ckme=ckme_1, es=state_space)
    sigma_2 = compute_sigmas(kernel=kernel.x, ckme=ckme_2, es=state_space)

    cmmds = kernel.y.distance.batch(kmes_1, kmes_2)

    return Result(
        dataset_1=dataset_1, dataset_2=dataset_2,
        cmmds=cmmds,
        beta_1_botstrap=beta_1_bootstrap, beta_2_bootstrap=beta_2_bootstrap,
        beta_1_analytical=beta_1_analytical, beta_2_analytical=beta_2_analytical,
        sigmas_1=sigma_1, sigmas_2=sigma_2,
        values_1=kernel.x.evaluate.one_many(function_1, state_space),
        values_2=kernel.x.evaluate.one_many(function_2, state_space),
        estimated_values_1=jax.vmap(jnp.dot)(kmes_1.coefficients, kmes_1.points),
        estimated_values_2=jax.vmap(jnp.dot)(kmes_2.coefficients, kmes_2.points),
    )


if __name__ == "__main__":
    set_plot_style()
    main(DEFAULT_CONFIG)
