from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
import tqdm
import tyro
from jax import Array

from src import expyro
from src.experiments.monitoring.plots import plot_thresholds, plot_cmmd, plot_ratio, plot_beta, \
    plot_online_trajectories, plot_posterior_std
from src.experiments.monitoring.spec import MultipleResult, Config, DEFAULT_ARGS, SingleResult
from src.rkhs import VectorKernel
from src.rkhs.testing import BootstrapConditionalTestEmbedding, two_sample_test, TestOutcome
from src.util import generate_random_keys, move_experiment_run, DIR_RESULTS, set_plot_style

PLOTS = [plot_ratio, plot_thresholds, plot_cmmd, plot_beta, plot_posterior_std, plot_online_trajectories]


def make_windows(trajectory: Array, window_size: int) -> tuple[Array, Array, Array]:
    assert trajectory.ndim == 2
    assert window_size >= 1

    xs = trajectory[:-1]
    ys = trajectory[1:]

    length_trajectory = trajectory.shape[0] - 1
    window_indices = jnp.arange(window_size) + jnp.arange(length_trajectory - window_size + 1)[:, None]

    xs_windows = xs[window_indices]
    ys_windows = ys[window_indices]

    return xs_windows, ys_windows, xs_windows


class Monitoring(NamedTuple):
    outcomes: TestOutcome
    betas: Array
    posterior_std_reference: Array
    posterior_std_windows: Array

    @classmethod
    def from_stream(
            cls,
            kernel_reference: VectorKernel, kernel_windows: VectorKernel,
            reference_dataset: Array, trajectory: Array,
            window_size: int, n_bootstrap: int, significance_level: float,
            key: Array
    ) -> Self:
        xs_reference = reference_dataset[:-1]
        ys_reference = reference_dataset[1:]

        key, key_cme_reference = jax.random.split(key)

        cme_reference = BootstrapConditionalTestEmbedding.from_data(
            kernel=kernel_reference, xs=xs_reference, ys=ys_reference, es=xs_reference,
            n_bootstrap=n_bootstrap, key=key_cme_reference
        )

        xs_windows, ys_windows, es_windows = make_windows(trajectory, window_size)

        def step(xs_window: Array, ys_window: Array, es_window: Array, key_window: Array) -> Monitoring:
            cme = BootstrapConditionalTestEmbedding.from_data(
                kernel_windows, xs_window, ys_window, xs_reference,
                n_bootstrap=n_bootstrap, key=key_window,
            )

            kmes_reference = cme_reference(es_window)
            kmes_window = cme(es_window)

            beta = kmes_window.beta(significance_level)
            posterior_std_reference = cme_reference.posterior_std(es_window)
            posterior_std_windows = cme.posterior_std(es_window)
            outcome = two_sample_test(kmes_reference, kmes_window, significance_level=significance_level)

            return Monitoring(
                outcomes=outcome,
                betas=beta,
                posterior_std_reference=posterior_std_reference,
                posterior_std_windows=posterior_std_windows,
            )

        n_steps = xs_windows.shape[0]
        keys = jax.random.split(key, n_steps)

        return jax.lax.map(
            f=lambda inp: step(*inp),
            xs=(xs_windows, ys_windows, es_windows, keys)
        )


@expyro.plot(*PLOTS, file_format="png")
@expyro.experiment(DIR_RESULTS, name="monitoring")
def experiment(config: Config) -> MultipleResult:
    rng = generate_random_keys(config.seed)

    kernel_reference, kernel_windows = config.vector_kernels()

    dynamics = config.sample_dynamics(next(rng))
    dynamics_disturbed = config.disturb_dynamics(dynamics, next(rng))

    runs = []

    for _ in tqdm.trange(config.n_repetitions):
        reference_dataset = config.sample_reference_dataset(dynamics=dynamics, key=next(rng))
        online_trajectory = config.sample_online_trajectory(dynamics, dynamics_disturbed, next(rng))

        monitoring = Monitoring.from_stream(
            kernel_reference=kernel_reference, kernel_windows=kernel_windows,
            reference_dataset=reference_dataset,
            trajectory=online_trajectory,
            window_size=config.window_size, n_bootstrap=config.test.n_bootstrap,
            significance_level=config.test.significance_level, key=next(rng)
        )

        run = SingleResult(
            outcomes=monitoring.outcomes,
            reference_dataset=reference_dataset,
            online_trajectory=online_trajectory,
            beta=monitoring.betas,
            posterior_std_reference=monitoring.posterior_std_reference,
            posterior_std_windows=monitoring.posterior_std_windows
        )

        runs.append(run)

    return MultipleResult(
        runs=runs,
        dynamics=dynamics,
        dynamics_disturbed=dynamics_disturbed
    )


def main(dimension: int, disturbance: float, seed: int):
    config = Config(
        **DEFAULT_ARGS,
        seed=seed,
        dim=dimension,
        disturbance=disturbance
    )

    sub_dir = f"d={dimension}__disturbance={disturbance}"
    run_name = f"seed={seed}"

    if (experiment.directory / experiment.name / sub_dir / run_name).exists():
        print(f"Skipping {sub_dir}/{run_name} because it already exists.")
        return

    run = experiment(config)

    move_experiment_run(
        run=run,
        sub_dir=sub_dir,
        dir_name=run_name
    )


if __name__ == "__main__":
    set_plot_style()
    tyro.cli(main)
