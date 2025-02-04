from functools import partial

import jax
import jax.numpy as jnp

from src import expyro
from src.bounds import bootstrap_cmmd_thresholds
from src.data import RKHSFnSampling
from src.experiments.monitoring.spec import Config, Result, DEFAULT_ARGS
from src.figures.util import set_plot_style
from src.random import generate_random_keys
from src.util import KernelParametrization, move_experiment_run, DIR_RESULTS


def monitor(
        kernel: KernelParametrization,
        dataset_reference: RKHSFnSampling,
        trajectory: RKHSFnSampling,
        window_size: int, n_evaluations: int,
        n_bootstrap: int, confidence_level: float,
        key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    length_trajectory = trajectory.xs.shape[0]
    window_indices = jnp.arange(window_size) + jnp.arange(length_trajectory - window_size + 1)[:, None]

    xs_windows = trajectory.xs[window_indices]
    ys_windows = trajectory.ys[window_indices]
    es_windows = xs_windows[:, -n_evaluations:]

    ckme_reference = kernel.x.ckme(dataset_reference.xs, dataset_reference.ys, kernel.regularization)

    @partial(jax.vmap)
    def step(xs: jnp.ndarray, ys: jnp.ndarray, es: jnp.ndarray, key_: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        ckme_window = kernel.x.ckme(xs, ys, kernel.regularization)
        kmes_window = kernel.x.condition.one_many(ckme_window, es)
        kmes_reference = kernel.x.condition.one_many(ckme_reference, es)

        cmmds = kernel.y.distance.batch(kmes_window, kmes_reference)

        thresholds = bootstrap_cmmd_thresholds(
            kernel_x=kernel.x, kernel_y=kernel.y,
            ckme_1=ckme_reference, ckme_2=ckme_window,
            es_1=es, es_2=es,
            n_bootstrap=n_bootstrap, power=confidence_level,
            single_beta=True,
            key=key_
        )

        return cmmds, thresholds

    keys = jax.random.split(key, length_trajectory - window_size + 1)
    return step(xs_windows, ys_windows, es_windows, keys)


@expyro.experiment(DIR_RESULTS, "monitoring")
def main(config: Config):
    rng = generate_random_keys(config.seed)

    kernel = config.kernel.make()

    true_fn = config.sample_true_fn(kernel.x, next(rng))
    disturbance_fn = config.sample_disturbance_fn(kernel.x, next(rng))
    disturbed_fn = true_fn + disturbance_fn

    dataset_reference = config.sample_reference_dataset(kernel.x, true_fn, next(rng))

    nominal_trajectory = config.sample_trajectory(
        kernel=kernel.x,
        fn=true_fn,
        x_init=jnp.array([2.5, 2.5]),
        length=5 * config.size_window,
        key=next(rng)
    )

    anomalous_trajectory = config.sample_trajectory(
        kernel=kernel.x,
        fn=disturbed_fn,
        x_init=nominal_trajectory.xs[-1],
        length=5 * config.size_window,
        key=next(rng)
    )

    trajectory = RKHSFnSampling(
        xs=jnp.concatenate([nominal_trajectory.xs, anomalous_trajectory.xs]),
        ys=jnp.concatenate([nominal_trajectory.ys, anomalous_trajectory.ys])
    )

    cmmd_windows, threshold_windows = monitor(
        kernel=kernel,
        dataset_reference=dataset_reference,
        trajectory=trajectory,
        window_size=config.size_window,
        n_evaluations=config.n_evaluations,
        n_bootstrap=config.n_bootstrap, confidence_level=config.confidence_level,
        key=next(rng)
    )

    return Result(cmmd_windows=cmmd_windows, threshold_windows=threshold_windows)


if __name__ == "__main__":
    set_plot_style()

    for disturbance in [0.25, 0.5, 0.75, 1]:
        run = main(Config(
            **DEFAULT_ARGS,
            disturbance=disturbance
        ))

        move_experiment_run(
            run,
            sub_dir=None,
            dir_name=f"disturbance-{disturbance}",
        )
