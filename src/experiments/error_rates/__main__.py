from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Annotated, Union

import jax.numpy as jnp
import tqdm
import tyro
from jax import Array

from src import expyro
from src.experiments.error_rates.plots import plot_positive_rate
from src.experiments.error_rates.spec import Config, DatasetPairSamplingSpec, DisturbedDatasetPairSamplingSpec, \
    WeightedDisturbedDatasetPairSamplingSpec
from src.experiments.error_rates.tests import BootstrapTest, AnalyticalTest, HuLeiTest
from src.expyro.experiment import Run
from src.spec import VectorKernelSpec, GaussianKernelSpec, LinearKernelSpec, RKHSFnSpec, GaussianNoiseSpec, \
    GaussianMixtureNoiseSpec, PolynomialKernelSpec, SpaceSpec
from src.util import generate_random_keys, DIR_RESULTS, move_experiment_run, set_plot_style

N_REPETITIONS = 100
N_BOOTSTRAP = 500
RESOLUTION = 1000


@expyro.plot(plot_positive_rate, file_format="png")
@expyro.experiment(DIR_RESULTS, name="error_rates")
def experiment(config: Config) -> Array:
    rng = generate_random_keys(config.seed)

    kernel = config.kernel.make()

    dataset_pairs = config.sample_dataset_pairs(kernel.x, next(rng))

    outcomes = []

    for pair in tqdm.tqdm(dataset_pairs, total=len(dataset_pairs)):
        es = jnp.concatenate([pair.dataset_1.xs, pair.dataset_2.xs], axis=0)

        if isinstance(config.test, BootstrapTest.Spec):
            test = BootstrapTest(
                kernel=kernel,
                dataset_1=pair.dataset_1, dataset_2=pair.dataset_2,
                es=es,
                spec=config.test,
                key=next(rng)
            )
        elif isinstance(config.test, AnalyticalTest.Spec):
            test = AnalyticalTest(
                kernel=kernel,
                dataset_1=pair.dataset_1, dataset_2=pair.dataset_2,
                es=es,
                spec=config.test,
            )
        elif isinstance(config.test, HuLeiTest.Spec):
            assert isinstance(config.datasets.noise_1, GaussianNoiseSpec)
            assert isinstance(config.datasets.noise_2, GaussianNoiseSpec)

            test = HuLeiTest(
                spec=config.test,
                fn_1=pair.fn_1, fn_2=pair.fn_2,
                noise_1=config.datasets.noise_1, noise_2=config.datasets.noise_2,
                dataset_1=pair.dataset_1, dataset_2=pair.dataset_2,
                key=next(rng)
            )
        else:
            raise ValueError(f"Unknown test specification: {config.test}")

        outcome = test(config.significance_levels)
        outcomes.append(outcome)

    return jnp.stack(outcomes)


@dataclass(frozen=True)
class Experiment(ABC):
    @classmethod
    def sub_dir_name(cls) -> str:
        return cls.__name__

    @classmethod
    def dir_base(cls) -> Path:
        return experiment.directory / experiment.name / cls.sub_dir_name()

    @classmethod
    def cli(cls, name: str, description: str):
        @tyro.conf.configure(tyro.conf.OmitArgPrefixes)
        def main(setting: cls, seed: int):
            setting(seed)

        return Annotated[main, tyro.conf.subcommand(name=name, description=description)]

    @classmethod
    def load_runs(cls, sub_dir: str) -> list[Run[Config, Array]]:
        dir_base = cls.dir_base() / sub_dir
        assert dir_base.exists()

        return [
            experiment[run_dir]
            for run_dir in dir_base.iterdir()
            if run_dir.is_dir()
        ]

    @abstractmethod
    def configs(self, seed: int) -> dict[str, Config]:
        raise NotImplementedError

    def __call__(self, seed: int):
        configs = self.configs(seed)
        run_name = f"seed={seed:03d}"

        for sub_dir, config in configs.items():
            target_dir = self.dir_base() / sub_dir / run_name

            if target_dir in experiment:
                print(f"Skipping {sub_dir} because it already exists.")
                continue

            run = experiment(config)

            if target_dir.exists():
                print(f"Skipping {sub_dir} because it already exists.")
                continue

            move_experiment_run(
                run=run, sub_dir=f"{self.sub_dir_name()}/{sub_dir}", dir_name=run_name, skip_exists=True
            )


@dataclass(frozen=True)
class GlobalDisturbanceExperiment(Experiment):
    test_name: Literal["bootstrap", "hu-lei__gt", "hu-lei__hg"]
    relative_norm: float

    @property
    def noise_std(self):
        return 0.1

    @property
    def kernel(self) -> VectorKernelSpec:
        return VectorKernelSpec(
            x=GaussianKernelSpec(bandwidth=0.25, ndim=1),
            y=LinearKernelSpec(ndim=0),
            regularization=0.1
        )

    @property
    def space(self) -> SpaceSpec:
        return SpaceSpec(dim=2, min=-1, max=1)

    @property
    def rkhs_fn(self) -> RKHSFnSpec:
        return RKHSFnSpec(n_basis_points=12, ball_radius=1)

    @property
    def noise(self) -> GaussianNoiseSpec:
        return GaussianNoiseSpec(mean=jnp.array([0]), std=jnp.array([self.noise_std]))

    @property
    def test(self) -> BootstrapTest.Spec | HuLeiTest.Spec:
        if self.test_name == "bootstrap":
            return BootstrapTest.Spec(n_bootstrap=N_BOOTSTRAP)
        elif self.test_name == "hu-lei__gt":
            return HuLeiTest.SpecGroundTruthDensity()
        elif self.test_name == "hu-lei__hg":
            return HuLeiTest.SpecHomoscedasticGaussian(
                kernel=self.kernel,
                p_train=0.5,
                noise_std=self.noise_std,
            )
        else:
            raise ValueError(f"Unknown test: {self.test_name}")

    @property
    def dataset_size(self) -> int:
        return 100

    @property
    def n_disturbance_basis_point(self) -> int:
        return self.rkhs_fn.n_basis_points

    def __post_init__(self):
        assert self.relative_norm >= 0

    def configs(self, seed: int) -> dict[str, Config]:
        return {
            f"{self.test_name}/H0": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=True,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.dataset_size,
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            ),
            f"{self.test_name}/H1/norm={self.relative_norm}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DisturbedDatasetPairSamplingSpec(
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.dataset_size,
                    n_disturbance_basis_point=self.n_disturbance_basis_point,
                    relative_disturbance_norm=self.relative_norm,
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            )
        }


@dataclass(frozen=True)
class LocalDisturbanceExperiment(Experiment):
    test_name: Literal["bootstrap", "hu-lei__gt", "hu-lei__hg"]
    relative_norm: float
    relative_norm: float
    weight: float
    tolerance: float

    @property
    def noise_std(self):
        return 0.025

    @property
    def bandwidth(self) -> float:
        return 0.25

    @property
    def kernel(self) -> VectorKernelSpec:
        return VectorKernelSpec(
            x=GaussianKernelSpec(bandwidth=self.bandwidth, ndim=1),
            y=LinearKernelSpec(ndim=0),
            regularization=0.1
        )

    @property
    def space(self) -> SpaceSpec:
        return SpaceSpec(dim=2, min=-3, max=3)

    @property
    def rkhs_fn(self) -> RKHSFnSpec:
        return RKHSFnSpec(n_basis_points=36, ball_radius=1)

    @property
    def noise(self) -> GaussianNoiseSpec:
        return GaussianNoiseSpec(mean=jnp.array([0]), std=jnp.array([self.noise_std]))

    @property
    def test(self) -> BootstrapTest.Spec | HuLeiTest.Spec:
        if self.test_name == "bootstrap":
            return BootstrapTest.Spec(n_bootstrap=N_BOOTSTRAP)
        elif self.test_name == "hu-lei__gt":
            return HuLeiTest.SpecGroundTruthDensity()
        elif self.test_name == "hu-lei__hg":
            return HuLeiTest.SpecHomoscedasticGaussian(
                kernel=self.kernel,
                p_train=0.5,
                noise_std=self.noise_std,
            )
        else:
            raise ValueError(f"Unknown test: {self.test_name}")

    @property
    def dataset_size(self) -> int:
        return 500

    def __post_init__(self):
        assert self.relative_norm >= 0
        assert self.tolerance > 0

    def configs(self, seed: int) -> dict[str, Config]:
        return {
            f"{self.test_name}/H0": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=True,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.dataset_size,
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            ),
            f"{self.test_name}/H1/weight={self.weight}__tol={self.tolerance}__norm={self.relative_norm}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=WeightedDisturbedDatasetPairSamplingSpec(
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.dataset_size,
                    relative_disturbance_norm=self.relative_norm,
                    tolerance=self.tolerance,
                    weight=self.weight
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            )
        }


@dataclass(frozen=True)
class BootstrapVsAnalyticalExperiment(Experiment):
    test_name: Literal["analytical", "bootstrap"]
    noise_std: float

    @property
    def kernel(self) -> VectorKernelSpec:
        return VectorKernelSpec(
            x=GaussianKernelSpec(bandwidth=0.25, ndim=1),
            y=LinearKernelSpec(ndim=0),
            regularization=0.25
        )

    @property
    def noise(self) -> GaussianNoiseSpec:
        return GaussianNoiseSpec(mean=jnp.array([0]), std=jnp.array([self.noise_std]))

    @property
    def space(self) -> SpaceSpec:
        return SpaceSpec(dim=2, min=-1, max=1)

    @property
    def rkhs_fn(self) -> RKHSFnSpec:
        return RKHSFnSpec(n_basis_points=12, ball_radius=1)

    @property
    def test(self) -> AnalyticalTest.Spec | BootstrapTest.Spec:
        if self.test_name == "analytical":
            return AnalyticalTest.Spec(
                rkhs_norm=self.rkhs_fn.ball_radius,
                sub_gaussian_std=self.noise.sub_gaussian_std()
            )
        elif self.test_name == "bootstrap":
            return BootstrapTest.Spec(n_bootstrap=N_BOOTSTRAP)
        else:
            raise ValueError(f"Unknown test: {self.test_name}")

    @property
    def dataset_size(self) -> int:
        return 100

    def configs(self, seed: int) -> dict[str, Config]:
        return {
            f"{self.test_name}/H0/noise_std={self.noise_std}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=True,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.dataset_size
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed,
            ),
            f"{self.test_name}/H1/noise_std={self.noise_std}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=False,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.dataset_size
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed,
            )
        }


@dataclass(frozen=True)
class DatasetSizeExperiment(Experiment):
    test_name: Literal["analytical", "bootstrap"]
    size: int

    @property
    def noise_std(self) -> float:
        return 0.2

    @property
    def noise(self) -> GaussianNoiseSpec:
        return GaussianNoiseSpec(mean=jnp.array([0]), std=jnp.array([self.noise_std]))

    @property
    def kernel(self) -> VectorKernelSpec:
        return VectorKernelSpec(
            x=GaussianKernelSpec(bandwidth=0.25, ndim=1),
            y=LinearKernelSpec(ndim=0),
            regularization=0.5
        )

    @property
    def space(self) -> SpaceSpec:
        return SpaceSpec(dim=2, min=-1, max=1)

    @property
    def rkhs_fn(self):
        return RKHSFnSpec(n_basis_points=12, ball_radius=1)

    @property
    def test(self) -> BootstrapTest.Spec | AnalyticalTest.Spec:
        if self.test_name == "analytical":
            return AnalyticalTest.Spec(rkhs_norm=self.rkhs_fn.ball_radius, sub_gaussian_std=self.noise_std)
        elif self.test_name == "bootstrap":
            return BootstrapTest.Spec(n_bootstrap=N_BOOTSTRAP)
        else:
            raise ValueError(f"Unknown test: {self.test_name}")

    def configs(self, seed: int) -> dict[str, Config]:
        return {
            f"{self.test_name}/H0/size={self.size}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=True,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.size
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            ),
            f"{self.test_name}/H1/size={self.size}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=False,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.size
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            )
        }


@dataclass(frozen=True)
class MixtureNoiseExperiment(Experiment):
    kernel_type: Literal["gaussian", "linear"]
    noise_mean: float

    @property
    def noise_std(self) -> float:
        return 0.025

    @property
    def noise_gaussian(self) -> GaussianNoiseSpec:
        return GaussianNoiseSpec(mean=jnp.array([0]), std=jnp.array([self.noise_std]))

    @property
    def noise_mixture(self) -> GaussianMixtureNoiseSpec:
        return GaussianMixtureNoiseSpec(
            mean=jnp.array([-self.noise_mean, self.noise_mean]),
            std=jnp.array([self.noise_std, self.noise_std])
        )

    @property
    def space(self) -> SpaceSpec:
        return SpaceSpec(dim=2, min=-1, max=1)

    @property
    def rkhs_fn(self) -> RKHSFnSpec:
        return RKHSFnSpec(n_basis_points=12, ball_radius=1)

    @property
    def kernel(self) -> VectorKernelSpec:
        if self.kernel_type == "gaussian":
            kernel_y = GaussianKernelSpec(bandwidth=0.05, ndim=0)
        elif self.kernel_type == "linear":
            kernel_y = LinearKernelSpec(ndim=0)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return VectorKernelSpec(
            x=GaussianKernelSpec(bandwidth=0.25, ndim=1),
            y=kernel_y,
            regularization=0.5
        )

    @property
    def dataset_size(self) -> int:
        return 100

    @property
    def test(self) -> BootstrapTest.Spec:
        return BootstrapTest.Spec(n_bootstrap=N_BOOTSTRAP)

    def __post_init__(self):
        assert self.noise_mean >= 0

    def configs(self, seed: int) -> dict[str, Config]:
        return {
            f"{self.kernel_type}/H0": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=True,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise_gaussian, noise_2=self.noise_gaussian,
                    dataset_size=self.dataset_size
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            ),
            f"{self.kernel_type}/H1/noise_mean={self.noise_mean}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=True,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise_gaussian, noise_2=self.noise_mixture,
                    dataset_size=self.dataset_size
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            )
        }


@dataclass(frozen=True)
class OutputKernelExperiment(Experiment):
    kernel_type: Literal["gaussian", "polynomial"]
    kernel_parameter: int | float

    @property
    def noise_std(self) -> float:
        return 0.2

    @property
    def noise(self) -> GaussianNoiseSpec:
        return GaussianNoiseSpec(mean=jnp.array([0]), std=jnp.array([self.noise_std]))

    @property
    def space(self) -> SpaceSpec:
        return SpaceSpec(dim=2, min=-1, max=1)

    @property
    def rkhs_fn(self) -> RKHSFnSpec:
        return RKHSFnSpec(n_basis_points=12, ball_radius=1)

    @property
    def kernel(self) -> VectorKernelSpec:
        if self.kernel_type == "gaussian":
            kernel_y = GaussianKernelSpec(bandwidth=self.kernel_parameter, ndim=0)
        elif self.kernel_type == "polynomial":
            kernel_y = PolynomialKernelSpec(degree=self.kernel_parameter, ndim=0)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return VectorKernelSpec(
            x=GaussianKernelSpec(bandwidth=0.25, ndim=1),
            y=kernel_y,
            regularization=0.2
        )

    @property
    def dataset_size(self) -> int:
        return 100

    @property
    def test(self) -> BootstrapTest.Spec:
        return BootstrapTest.Spec(n_bootstrap=N_BOOTSTRAP)

    def __post_init__(self):
        if self.kernel_type == "gaussian":
            assert isinstance(self.kernel_parameter, float | int)
        elif self.kernel_type == "polynomial":
            assert isinstance(self.kernel_parameter, int)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def configs(self, seed: int) -> dict[str, Config]:
        return {
            f"{self.kernel_type}/H0/parameter={self.kernel_parameter}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=True,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.dataset_size
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            ),
            f"{self.kernel_type}/H1/parameter={self.kernel_parameter}": Config(
                kernel=self.kernel,
                space=self.space,
                datasets=DatasetPairSamplingSpec(
                    single_fn=False,
                    rkhs_fn=self.rkhs_fn,
                    noise_1=self.noise, noise_2=self.noise,
                    dataset_size=self.dataset_size
                ),
                test=self.test,
                resolution=RESOLUTION,
                n_repetitions=N_REPETITIONS,
                seed=seed
            )
        }


if __name__ == "__main__":
    set_plot_style()

    tyro.cli(
        Union[
            BootstrapVsAnalyticalExperiment.cli(
                name="bootstrap-vs-analytical",
                description="compare performance for bootstrapped and analytical thresholds"
            ),
            DatasetSizeExperiment.cli(
                name="dataset-size",
                description="compare performance for different dataset sizes"
            ),
            GlobalDisturbanceExperiment.cli(
                name="disturbance",
                description="compare performance for different levels of disturbance to a ground-truth mean function"
            ),
            LocalDisturbanceExperiment.cli(
                name="local-disturbance",
                description="compare performance for different probabilities of sampling close to a disturbance"
            ),
            MixtureNoiseExperiment.cli(
                name="mixture-noise",
                description="compare performance for differently far apart mixture of Gaussians noise distributions"
            ),
            OutputKernelExperiment.cli(
                name="output-kernel",
                description="compare performance for different output kernels"
            )
        ]
    )
