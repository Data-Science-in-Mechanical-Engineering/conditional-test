from functools import partial

import jax
import jax.numpy as jnp

from src import expyro
from src.config import ThresholdConfig, GaussianMixtureNoiseConfig
from src.experiments.pairwise_pointwise.plots import plot_data, plot_thresholds, plot_true_cmmd, \
    plot_cmmd_estimate, plot_rejection_region
from src.experiments.pairwise_pointwise.spec import MeanConfig, DistributionalConfig, Result, CONFIG_MEAN_IID, \
    CONFIG_DISTRIBUTIONAL_IID
from src.random import generate_random_keys
from src.rkhs import CKME, RKHSFn
from src.util import DIR_RESULTS, KernelParametrization

DIR = DIR_RESULTS / "pairwise_pointwise"


def estimate_cmmd_grid_pairwise(kernel: KernelParametrization, ckmes_1: CKME, ckmes_2: CKME,
                                es: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.map(lambda ckme_1: jax.lax.map(
        lambda ckme_2: kernel.y.distance.batch(
            kernel.x.condition.one_many(ckme_1, es),
            kernel.x.condition.one_many(ckme_2, es)
        ), ckmes_2
    ), ckmes_1)


def compute_cmmd_threshold_pairwise(
        threshold: ThresholdConfig,
        kernel: KernelParametrization,
        ckmes_1: CKME, ckmes_2: CKME,
        es: jnp.ndarray,
        rkhs_norm_1: float, rkhs_norm_2: float,
        sub_gaussian_std_1: float, sub_gaussian_std_2: float,
        power: float,
        key: jax.Array
) -> jnp.ndarray:
    return jax.lax.map(
        lambda ckme_1: jax.lax.map(
            lambda ckme_2: threshold.thresholds(
                kernel=kernel,
                ckme_1=ckme_1, ckme_2=ckme_2,
                es=es,
                rkhs_norm_1=rkhs_norm_1, rkhs_norm_2=rkhs_norm_2,
                sub_gaussian_std_1=sub_gaussian_std_1, sub_gaussian_std_2=sub_gaussian_std_2,
                power=power,
                key=key
            ), ckmes_2
        ), ckmes_1
    )


@expyro.plot(plot_data, file_format="png")
@expyro.plot(plot_thresholds, file_format="png")
@expyro.plot(plot_rejection_region, file_format="png")
@expyro.plot(plot_cmmd_estimate, file_format="png")
@expyro.plot(plot_true_cmmd, file_format="png")
@expyro.experiment(DIR)
def pairwise_mean_comparison(config: MeanConfig):
    rng = generate_random_keys(config.seed)

    kernel = config.parametrization().make()
    state_space = config.data.space.discretization(config.resolution)

    fns = config.sample_rkhs_fns(kernel.x, next(rng))
    datasets_1 = config.sample_noisy_datasets(kernel.x, fns, config.noise_1_top, config.noise_1_bottom, next(rng))
    datasets_2 = config.sample_noisy_datasets(kernel.x, fns, config.noise_2_top, config.noise_2_bottom, next(rng))

    ckmes_1 = kernel.x.ckmes(datasets_1.xs, datasets_1.ys, kernel.regularization)
    ckmes_2 = kernel.x.ckmes(datasets_2.xs, datasets_2.ys, kernel.regularization)

    cmmd_grid_pairwise = estimate_cmmd_grid_pairwise(kernel, ckmes_1, ckmes_2, state_space).reshape(
        config.n_functions, config.n_functions, config.resolution, config.resolution
    )

    fns_value_grid = kernel.x.evaluate.many_many(fns, state_space)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def cmmd_ground_truth_pairwise(fn_values_1: jnp.ndarray, fn_values_2: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(fn_values_1 - fn_values_2).reshape(config.resolution, config.resolution)

    true_cmmd_pairwise = cmmd_ground_truth_pairwise(fns_value_grid, fns_value_grid)

    sub_gaussian_std = max(
        config.noise_1_top.sub_gaussian_std(),
        config.noise_1_bottom.sub_gaussian_std(),
        config.noise_2_top.sub_gaussian_std(),
        config.noise_2_bottom.sub_gaussian_std()
    )

    threshold_grid_pairwise = compute_cmmd_threshold_pairwise(
        threshold=config.threshold,
        kernel=kernel,
        ckmes_1=ckmes_1, ckmes_2=ckmes_2,
        es=state_space,
        rkhs_norm_1=config.rkhs_ball_radius, rkhs_norm_2=config.rkhs_ball_radius,
        sub_gaussian_std_1=sub_gaussian_std, sub_gaussian_std_2=sub_gaussian_std,
        power=config.threshold.confidence_level,
        key=next(rng)
    ).reshape(config.n_functions, config.n_functions, config.resolution, config.resolution)

    rejection_grid_pairwise = jnp.astype(cmmd_grid_pairwise > threshold_grid_pairwise, jnp.int32)

    return Result(
        functions=fns,
        datasets_1=datasets_1,
        datasets_2=datasets_2,
        cmmd_grid_pairwise=cmmd_grid_pairwise,
        fn_values_grid_pairwise=fns_value_grid,
        true_cmmd_grid_pairwise=true_cmmd_pairwise,
        threshold_grid_pairwise=threshold_grid_pairwise,
        rejection_grid_pairwise=rejection_grid_pairwise
    )


@expyro.plot(plot_data, file_format="png")
@expyro.plot(plot_thresholds, file_format="png")
@expyro.plot(plot_rejection_region, file_format="png")
@expyro.plot(plot_cmmd_estimate, file_format="png")
@expyro.plot(plot_true_cmmd, file_format="png")
@expyro.experiment(DIR)
def pairwise_distributional_comparison(config: DistributionalConfig):
    rng = generate_random_keys(config.seed)

    kernel = config.parametrization().make()
    state_space = config.data.space.discretization(config.resolution)

    fns = config.sample_rkhs_fns(kernel.x, next(rng))
    datasets_1 = config.sample_noisy_datasets(kernel.x, fns, config.noise_1_top, config.noise_1_bottom, next(rng))
    datasets_2 = config.sample_noisy_datasets(kernel.x, fns, config.noise_2_top, config.noise_2_bottom, next(rng))

    ckmes_1 = kernel.x.ckmes(datasets_1.xs, datasets_1.ys, kernel.regularization)
    ckmes_2 = kernel.x.ckmes(datasets_2.xs, datasets_2.ys, kernel.regularization)

    cmmd_grid_pairwise = estimate_cmmd_grid_pairwise(kernel, ckmes_1, ckmes_2, state_space).reshape(
        config.n_functions, config.n_functions, config.resolution, config.resolution
    )

    threshold_grid_pairwise = compute_cmmd_threshold_pairwise(
        threshold=config.threshold,
        kernel=kernel,
        ckmes_1=ckmes_1, ckmes_2=ckmes_2,
        es=state_space,
        rkhs_norm_1=-1, rkhs_norm_2=-1,
        sub_gaussian_std_1=-1, sub_gaussian_std_2=-1,
        power=config.threshold.confidence_level,
        key=next(rng)
    ).reshape(config.n_functions, config.n_functions, config.resolution, config.resolution)

    rejection_grid_pairwise = jnp.astype(cmmd_grid_pairwise > threshold_grid_pairwise, jnp.int32)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def cmmd_ground_truth_pairwise(fn_1: RKHSFn, fn_2: RKHSFn) -> jnp.ndarray:
        cmmd_top = GaussianMixtureNoiseConfig.closed_form_mmd_batch(
            noise_1=config.noise_1_top, noise_2=config.noise_2_top,
            kernel_x=kernel.x, kernel_y=kernel.y,
            fn_1=fn_1,
            fn_2=fn_2,
            xs=state_space
        )

        cmmd_bottom = GaussianMixtureNoiseConfig.closed_form_mmd_batch(
            noise_1=config.noise_1_bottom, noise_2=config.noise_2_bottom,
            kernel_x=kernel.x, kernel_y=kernel.y,
            fn_1=fn_1,
            fn_2=fn_2,
            xs=state_space
        )

        mask = config.state_space_noise_mask(state_space)

        cmmd = cmmd_top * mask + cmmd_bottom * ~mask

        return cmmd.reshape(config.resolution, config.resolution)

    true_cmmd_pairwise = cmmd_ground_truth_pairwise(fns, fns)

    return Result(
        functions=fns,
        datasets_1=datasets_1,
        datasets_2=datasets_2,
        cmmd_grid_pairwise=cmmd_grid_pairwise,
        fn_values_grid_pairwise=kernel.x.evaluate.many_many(fns, state_space),
        true_cmmd_grid_pairwise=true_cmmd_pairwise,
        threshold_grid_pairwise=threshold_grid_pairwise,
        rejection_grid_pairwise=rejection_grid_pairwise
    )


if __name__ == "__main__":
    pairwise_mean_comparison(CONFIG_MEAN_IID)
    pairwise_distributional_comparison(CONFIG_DISTRIBUTIONAL_IID)
