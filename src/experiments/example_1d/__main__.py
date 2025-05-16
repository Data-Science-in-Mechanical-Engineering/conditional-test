import tqdm

from src import expyro
from src.experiments.example_1d.plots import plot_curves
from src.experiments.example_1d.spec import Config, DEFAULT_ARGS, Result
from src.rkhs.testing import BootstrapConditionalTestEmbedding, AnalyticalConditionalTestEmbedding, two_sample_test
from src.spec import AnalyticalTestSpec, BootstrapTestSpec
from src.util import generate_random_keys, set_plot_style, DIR_RESULTS, move_experiment_run


@expyro.plot(plot_curves, file_format="png")
@expyro.experiment(DIR_RESULTS, name="example_1d")
def experiment(config: Config) -> Result:
    rng = generate_random_keys(config.seed)

    kernel = config.make_kernel()
    state_space = config.SPACE.discretization(config.resolution)

    function_1 = config.sample_rkhs_fn(kernel, next(rng))
    function_2 = config.sample_rkhs_fn(kernel, next(rng))

    dataset_1 = config.sample_dataset(function_1, next(rng))
    dataset_2 = config.sample_dataset(function_2, next(rng))

    if isinstance(config.test, AnalyticalTestSpec):
        sub_gaussian_std = config.sub_gaussian_std()

        cme_1 = AnalyticalConditionalTestEmbedding.from_data(
            kernel, dataset_1.xs, dataset_1.ys, rkhs_norm=config.rkhs_fn.ball_radius, sub_gaussian_std=sub_gaussian_std
        )

        cme_2 = AnalyticalConditionalTestEmbedding.from_data(
            kernel, dataset_2.xs, dataset_2.ys, rkhs_norm=config.rkhs_fn.ball_radius, sub_gaussian_std=sub_gaussian_std
        )
    elif isinstance(config.test, BootstrapTestSpec):
        cme_1 = BootstrapConditionalTestEmbedding.from_data(
            kernel, dataset_1.xs, dataset_1.ys, dataset_1.xs, n_bootstrap=config.test.n_bootstrap, key=next(rng)
        )

        cme_2 = BootstrapConditionalTestEmbedding.from_data(
            kernel, dataset_2.xs, dataset_2.ys, dataset_2.xs, n_bootstrap=config.test.n_bootstrap, key=next(rng)
        )
    else:
        raise ValueError(f"Unknown test: {config.test}")

    kmes_1 = cme_1(state_space)
    kmes_2 = cme_2(state_space)

    result = two_sample_test(kmes_1, kmes_2, config.test.significance_level)

    return Result(
        results=result, fn_1=function_1, fn_2=function_2, cme_1=cme_1, cme_2=cme_2, kmes_1=kmes_1, kmes_2=kmes_2
    )


def main():
    set_plot_style()

    tests = {
        "analytical": AnalyticalTestSpec(significance_level=0.05),
        "bootstrap": BootstrapTestSpec(significance_level=0.05, n_bootstrap=1000)
    }

    for name, test in tqdm.tqdm(tests.items()):
        config = Config(**DEFAULT_ARGS, test=test)
        run = experiment(config)
        move_experiment_run(run, DIR_RESULTS / "example_1d", name)


if __name__ == "__main__":
    main()
