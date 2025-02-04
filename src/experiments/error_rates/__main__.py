from functools import partial

import jax
import jax.numpy as jnp

from src import expyro
from src.config import ThresholdConfig, KernelParametrizationConfig, GaussianKernelConfig, IIDDataConfig, \
    GaussianMixtureNoiseConfig, LinearKernelConfig, PolynomialKernelConfig, KernelConfig, AnalyticalThresholdConfig, \
    DataConfig, GaussianNoiseConfig
from src.data import RKHSFnSampling
from src.experiments.error_rates.plots import plot_positive_rates
from src.experiments.error_rates.spec import PositiveRate, BaseConfig, SingleFnDifferentNoiseConfig, DEFAULT_ARGS, \
    DEFAULT_BOOTSTRAP, SingleFnSingleNoiseConfig, DifferentFnSingleNoiseConfig, \
    DEFAULT_MAX_CONFIDENCE_LEVEL, DEFAULT_SPACE, DEFAULT_IID_DATA, DisturbedFnSingleNoiseConfig
from src.figures.util import set_plot_style
from src.random import generate_random_keys
from src.util import KernelParametrization, DIR_RESULTS, move_experiment_run


def compute_rejection_rates_from_regions(
        rejection_regions: jnp.ndarray,
        n_bootstrap: int,
        key: jax.Array
) -> PositiveRate:
    def uniform_positive_rate(rejections: jnp.ndarray) -> jnp.ndarray:
        return (rejections > 0).any(axis=-1).mean(axis=-1)

    def local_positive_rate(rejections: jnp.ndarray) -> jnp.ndarray:
        return rejections.mean(axis=(-1, -2))

    def bootstrap(rejection_region: jnp.ndarray, key_: jax.Array) -> jnp.ndarray:
        return jax.random.choice(key_, rejection_region, (n_bootstrap, rejection_region.shape[0],), replace=True)

    def bootstrap_batch_fn(rejection_region_batch: jnp.ndarray, key_: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        keys_ = jax.random.split(key_, rejection_region_batch.shape[0])

        bootstrap_batch = jax.lax.map(lambda x: bootstrap(*x), xs=(rejection_region_batch, keys_))
        bootstrap_batch = bootstrap_batch.transpose(1, 0, 2)

        bootstrap_uniform_rate = uniform_positive_rate(bootstrap_batch)
        bootstrap_local_rate = local_positive_rate(bootstrap_batch)

        return bootstrap_uniform_rate, bootstrap_local_rate

    keys = jax.random.split(key, rejection_regions.shape[0])

    bootstrap_uniform_rate_distribution, bootstrap_local_rate_distribution = jax.lax.map(
        f=lambda x: bootstrap_batch_fn(*x),
        xs=(rejection_regions, keys)
    )

    return PositiveRate(
        uniform=uniform_positive_rate(rejection_regions),
        local=local_positive_rate(rejection_regions),
        bootstrap_uniform_distribution=bootstrap_uniform_rate_distribution,
        bootstrap_local_distribution=bootstrap_local_rate_distribution
    )


@partial(jax.jit, static_argnums={0, 1, 8})
def compute_rejection_rates(
        kernel: KernelParametrization,
        threshold_config: ThresholdConfig,
        dataset_batch_1: RKHSFnSampling, dataset_batch_2: RKHSFnSampling,
        state_space: jnp.ndarray,
        rkhs_norm: float,
        sub_gaussian_std: float,
        power_levels: jnp.ndarray,
        n_bootstrap: int,
        key: jax.Array
) -> PositiveRate:
    assert dataset_batch_1.xs.shape[0] == dataset_batch_2.xs.shape[0]
    assert dataset_batch_1.xs.shape[1] == dataset_batch_2.xs.shape[1]

    key_rejection, key_bootstrap = jax.random.split(key)

    rejection_regions = jax.lax.map(lambda datasets: compute_rejection_region(
        kernel=kernel,
        threshold_config=threshold_config,
        dataset_1=datasets[0], dataset_2=datasets[1],
        state_space=state_space,
        rkhs_norm=rkhs_norm,
        sub_gaussian_std=sub_gaussian_std,
        powers=power_levels,
        key=key_rejection
    ), xs=(dataset_batch_1, dataset_batch_2)).transpose(1, 0, 2)

    return compute_rejection_rates_from_regions(rejection_regions, n_bootstrap, key_bootstrap)


def compute_rejection_region(
        kernel: KernelParametrization,
        threshold_config: ThresholdConfig,
        dataset_1: RKHSFnSampling, dataset_2: RKHSFnSampling,
        state_space: jnp.ndarray,
        rkhs_norm: float,
        sub_gaussian_std: float,
        powers: jnp.ndarray,
        key: jax.Array
):
    assert powers.ndim == 1

    ckme_1 = kernel.x.ckme(dataset_1.xs, dataset_1.ys, kernel.regularization)
    ckme_2 = kernel.x.ckme(dataset_2.xs, dataset_2.ys, kernel.regularization)

    kme_batch_1 = kernel.x.condition.one_many(ckme_1, state_space)
    kme_batch_2 = kernel.x.condition.one_many(ckme_2, state_space)

    cmmd_batch = kernel.y.distance.batch(kme_batch_1, kme_batch_2)

    @partial(jax.vmap)
    def threshold_batch_fn(power: jnp.ndarray) -> jnp.ndarray:
        return threshold_config.thresholds(
            kernel=kernel,
            ckme_1=ckme_1, ckme_2=ckme_2,
            es=state_space,
            rkhs_norm_1=rkhs_norm, rkhs_norm_2=rkhs_norm,
            sub_gaussian_std_1=sub_gaussian_std, sub_gaussian_std_2=sub_gaussian_std,
            power=power,
            key=key
        )

    thresholds = threshold_batch_fn(powers)

    return cmmd_batch > thresholds


@expyro.plot(plot_positive_rates, file_format="png")
@expyro.experiment(DIR_RESULTS, "error_rates")
def experiment(config: BaseConfig) -> PositiveRate:
    rng = generate_random_keys(config.seed)

    kernel = config.kernel.make()

    confidence_levels = config.confidence_levels()
    evaluation_points = config.data.space.uniform_sample(config.n_evaluation_points, next(rng))

    fn_1, fn_2 = config.sample_rkhs_fn_pair(kernel.x, next(rng))
    dataset_1, dataset_2 = config.sample_dataset_pair(kernel.x, fn_1, fn_2, next(rng))

    return compute_rejection_rates(
        kernel=kernel,
        threshold_config=config.threshold,
        dataset_batch_1=dataset_1, dataset_batch_2=dataset_2,
        state_space=evaluation_points,
        rkhs_norm=config.rkhs_ball_radius,
        sub_gaussian_std=config.sub_gaussian_std(),
        power_levels=confidence_levels,
        n_bootstrap=config.n_bootstrap_rate,
        key=next(rng)
    )


def bootstrap_vs_analytical_experiment(data: DataConfig):
    for std in [0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        args = dict(
            **DEFAULT_ARGS,
            kernel=KernelParametrizationConfig(
                x_config=GaussianKernelConfig(bandwidth=0.5, ndim=1),
                y_config=LinearKernelConfig(ndim=0),
                regularization=0.5,
            ),
            data=data,
            noise=GaussianMixtureNoiseConfig(
                mean=jnp.array([0.0]),
                std=jnp.array([std]),
            )
        )

        config = SingleFnSingleNoiseConfig(
            **args,
            threshold=DEFAULT_BOOTSTRAP
        )

        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"bootstrap-vs-analytical/{data.__class__.__name__}",
            dir_name=f"same-fn__bootstrap__std-{std}"
        )

        config = DifferentFnSingleNoiseConfig(
            **args,
            threshold=DEFAULT_BOOTSTRAP,
        )

        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"bootstrap-vs-analytical/{data.__class__.__name__}",
            dir_name=f"different-fn__bootstrap__std-{std}"
        )

        config = SingleFnSingleNoiseConfig(
            **args,
            threshold=AnalyticalThresholdConfig(confidence_level=DEFAULT_MAX_CONFIDENCE_LEVEL)
        )

        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"bootstrap-vs-analytical/{data.__class__.__name__}",
            dir_name=f"same-fn__analytical__std-{std}"
        )

        config = DifferentFnSingleNoiseConfig(
            **args,
            threshold=AnalyticalThresholdConfig(confidence_level=DEFAULT_MAX_CONFIDENCE_LEVEL)
        )

        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"bootstrap-vs-analytical/{data.__class__.__name__}",
            dir_name=f"different-fn__analytical__std-{std}"
        )


def mixture_noise_intensity_experiment(data: DataConfig):
    for mean in [0.025, 0.05, 0.075, 0.1]:
        for kernel_y in [GaussianKernelConfig(bandwidth=0.05, ndim=0), LinearKernelConfig(ndim=0)]:
            args = dict(
                **DEFAULT_ARGS,
                kernel=KernelParametrizationConfig(
                    x_config=GaussianKernelConfig(bandwidth=0.5, ndim=1),
                    y_config=kernel_y,
                    regularization=0.5
                ),
                threshold=DEFAULT_BOOTSTRAP,
                data=data,
            )

            config = SingleFnDifferentNoiseConfig(
                **args,
                noise_1=GaussianMixtureNoiseConfig(
                    mean=jnp.array([0.0]),
                    std=jnp.array([0.01]),
                ),
                noise_2=GaussianMixtureNoiseConfig(
                    mean=jnp.array([-mean, mean]),
                    std=jnp.array([0.01, 0.01]),
                )
            )

            run = experiment(config)

            move_experiment_run(
                run,
                sub_dir=f"mixture-noise-intensity/{data.__class__.__name__}",
                dir_name=f"different-noise__mean-{mean}__kernel-y-{kernel_y.__class__.__name__}"
            )

            config = SingleFnSingleNoiseConfig(
                **args,
                noise=GaussianMixtureNoiseConfig(
                    mean=jnp.array([-mean, mean]),
                    std=jnp.array([0.01, 0.01]),
                )
            )

            run = experiment(config)

            move_experiment_run(
                run,
                sub_dir=f"mixture-noise-intensity/{data.__class__.__name__}",
                dir_name=f"same-noise__mean-{mean}__kernel-y-{kernel_y.__class__.__name__}"
            )


def moment_richness_experiment(data: DataConfig):
    noise_config = GaussianMixtureNoiseConfig(mean=jnp.array([0.0]), std=jnp.array([0.2]))

    def make_args(kernel_config: KernelConfig) -> dict:
        return dict(
            **DEFAULT_ARGS,
            kernel=KernelParametrizationConfig(
                x_config=GaussianKernelConfig(bandwidth=0.5, ndim=1),
                y_config=kernel_config,
                regularization=0.1,
            ),
            threshold=DEFAULT_BOOTSTRAP,
            data=data,
            noise=noise_config
        )

    for degree in [1, 2, 3]:
        kernel_y = PolynomialKernelConfig(degree=degree, ndim=0)

        args = make_args(kernel_y)
        config = SingleFnSingleNoiseConfig(**args)

        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"moment-richness/{data.__class__.__name__}",
            dir_name=f"same-fn__polynomial__degree-{degree}"
        )

        config = DifferentFnSingleNoiseConfig(**args)

        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"moment-richness/{data.__class__.__name__}",
            dir_name=f"different-fn__polynomial__degree-{degree}"
        )

    for bandwidth in [0.05, 0.1, 0.15]:
        kernel_y = GaussianKernelConfig(bandwidth=bandwidth, ndim=0)

        args = make_args(kernel_y)
        config = SingleFnSingleNoiseConfig(**args)

        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"moment-richness/{data.__class__.__name__}",
            dir_name=f"same-fn__gaussian__bandwidth-{bandwidth}"
        )

        config = DifferentFnSingleNoiseConfig(**args)

        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"moment-richness/{data.__class__.__name__}",
            dir_name=f"different-fn__gaussian__bandwidth-{bandwidth}"
        )


def dataset_size():
    for size in [10, 20, 50, 100]:
        args = dict(
            **DEFAULT_ARGS,
            kernel=KernelParametrizationConfig(
                x_config=GaussianKernelConfig(bandwidth=0.5, ndim=1),
                y_config=GaussianKernelConfig(bandwidth=0.2, ndim=0),
                regularization=0.5,
            ),
            threshold=DEFAULT_BOOTSTRAP,
            data=IIDDataConfig(space=DEFAULT_SPACE, dataset_size=size),
            noise=GaussianNoiseConfig(mean=jnp.array([0.0]), std=jnp.array([0.2]))
        )

        config = SingleFnSingleNoiseConfig(**args)
        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"dataset-size",
            dir_name=f"same-fn__size-{size}"
        )

        config = DifferentFnSingleNoiseConfig(**args)
        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"dataset-size",
            dir_name=f"different-fn__size-{size}"
        )


def disturbance(data: DataConfig):
    args = dict(
        **DEFAULT_ARGS,
        kernel=KernelParametrizationConfig(
            x_config=GaussianKernelConfig(bandwidth=0.5, ndim=1),
            y_config=GaussianKernelConfig(bandwidth=0.2, ndim=0),
            regularization=0.5,
        ),
        threshold=DEFAULT_BOOTSTRAP,
        data=data,
        noise=GaussianNoiseConfig(mean=jnp.array([0.0]), std=jnp.array([0.2]))
    )

    config = SingleFnSingleNoiseConfig(**args)
    run = experiment(config)

    move_experiment_run(
        run,
        sub_dir=f"disturbance",
        dir_name=f"single-fn"
    )

    for magnitude in [0.1, 0.25, 0.5, 0.75, 1]:
        config = DisturbedFnSingleNoiseConfig(**args, disturbance=magnitude)
        run = experiment(config)

        move_experiment_run(
            run,
            sub_dir=f"disturbance",
            dir_name=f"disturbed-fn__magnitude-{magnitude}"
        )


if __name__ == "__main__":
    set_plot_style()

    mixture_noise_intensity_experiment(DEFAULT_IID_DATA)
    moment_richness_experiment(DEFAULT_IID_DATA)
    bootstrap_vs_analytical_experiment(DEFAULT_IID_DATA)
    dataset_size()
    disturbance(DEFAULT_IID_DATA)
