import jax
import jax.numpy as jnp

from src import expyro
from src.config import BootstrapThresholdConfig
from src.data import RKHSFnSampling
from src.experiments.example_2d.spec import Config, DEFAULT_CONFIG, Result
from src.random import generate_random_keys
from src.util import KernelParametrization, DIR_RESULTS, move_experiment_run


def _compute_rejection_region(
        kernel: KernelParametrization, threshold_config: BootstrapThresholdConfig,
        state_space: jnp.ndarray,
        dataset_1: RKHSFnSampling, dataset_2: RKHSFnSampling,
        key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ckme_1 = kernel.x.ckme(dataset_1.xs, dataset_1.ys, kernel.regularization)
    ckme_2 = kernel.x.ckme(dataset_2.xs, dataset_2.ys, kernel.regularization)

    kmes_1 = kernel.x.condition.one_many(ckme_1, state_space)
    kmes_2 = kernel.x.condition.one_many(ckme_2, state_space)

    cmmds = kernel.y.distance.batch(kmes_1, kmes_2)

    thresholds = threshold_config.thresholds(
        kernel=kernel,
        ckme_1=ckme_1, ckme_2=ckme_2,
        es=state_space,
        rkhs_norm_1=-1, rkhs_norm_2=-1,
        sub_gaussian_std_1=-1, sub_gaussian_std_2=-1,
        power=threshold_config.confidence_level,
        key=key
    )

    return cmmds, thresholds


@expyro.experiment(DIR_RESULTS, "example_2d")
def main(config: Config):
    rng = generate_random_keys(config.seed)

    kernel = config.kernel.make()
    state_space = config.state_space()

    fn_1, fn_2 = config.sample_rkhs_fns(kernel.x, next(rng))

    dataset_iid_1 = config.sample_noisy_iid_dataset(kernel.x, fn_1, next(rng))
    dataset_iid_2 = config.sample_noisy_iid_dataset(kernel.x, fn_2, next(rng))

    dataset_rotation_1 = config.sample_noisy_rotation_dataset(kernel.x, fn_1, next(rng))
    dataset_rotation_2 = config.sample_noisy_rotation_dataset(kernel.x, fn_2, next(rng))

    cmmds_iid, thresholds_iid = _compute_rejection_region(
        kernel=kernel, threshold_config=config.threshold,
        state_space=state_space,
        dataset_1=dataset_iid_1, dataset_2=dataset_iid_2,
        key=next(rng)
    )

    cmmds_rotation, thresholds_rotation = _compute_rejection_region(
        kernel=kernel, threshold_config=config.threshold,
        state_space=state_space,
        dataset_1=dataset_rotation_1, dataset_2=dataset_rotation_2,
        key=next(rng)
    )

    return Result(
        cmmd_iid=cmmds_iid, cmmd_rotation=cmmds_rotation,
        thresholds_iid=thresholds_iid, thresholds_rotation=thresholds_rotation
    )


if __name__ == "__main__":
    run = main(DEFAULT_CONFIG)

    move_experiment_run(
        run=run,
        sub_dir=None,
        dir_name="res-1"
    )
