import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from src.data import RKHSFnSampling
from src.rkhs import VectorKernel, RKHSFn, CME
from src.rkhs.testing import ConditionedTestEmbedding, ConditionalTestEmbedding, two_sample_test, \
    AnalyticalConditionalTestEmbedding, BootstrapConditionalTestEmbedding
from src.spec import GaussianNoiseSpec, VectorKernelSpec, LinearKernelSpec


class Test(ABC):
    @final
    def __call__(self, significance_level: ArrayLike) -> Array:
        rejection = self._reject(significance_level)
        assert rejection.dtype == bool
        assert rejection.shape == (len(significance_level),)
        return rejection

    @abstractmethod
    def _reject(self, significance_level: ArrayLike) -> Array:
        raise NotImplementedError


class OurTest(Test):
    kmes_1: Final[ConditionedTestEmbedding]
    kmes_2: Final[ConditionedTestEmbedding]

    def __init__(self, cme_1: ConditionalTestEmbedding, cme_2: ConditionalTestEmbedding, es: Array):
        self.kmes_1 = jax.lax.map(cme_1, es)
        self.kmes_2 = jax.lax.map(cme_2, es)

    def _reject(self, significance_level: ArrayLike) -> Array:
        outcome = two_sample_test(self.kmes_1, self.kmes_2, significance_level)
        rejection = outcome.rejection()
        return rejection.any(axis=range(1, rejection.ndim))


class AnalyticalTest(OurTest):
    @dataclass(frozen=True)
    class Spec:
        rkhs_norm: float
        sub_gaussian_std: float

        def __post_init__(self):
            assert self.rkhs_norm >= 0
            assert self.sub_gaussian_std >= 0

    def __init__(
            self,
            kernel: VectorKernel,
            dataset_1: RKHSFnSampling, dataset_2: RKHSFnSampling,
            es: Array,
            spec: Spec
    ):
        cme_1 = AnalyticalConditionalTestEmbedding.from_data(
            kernel, dataset_1.xs, dataset_1.ys, rkhs_norm=spec.rkhs_norm, sub_gaussian_std=spec.sub_gaussian_std
        )

        cme_2 = AnalyticalConditionalTestEmbedding.from_data(
            kernel, dataset_2.xs, dataset_2.ys, rkhs_norm=spec.rkhs_norm, sub_gaussian_std=spec.sub_gaussian_std
        )

        super().__init__(cme_1, cme_2, es)


class BootstrapTest(OurTest):
    @dataclass(frozen=True)
    class Spec:
        n_bootstrap: int

    def __init__(
            self,
            kernel: VectorKernel,
            dataset_1: RKHSFnSampling, dataset_2: RKHSFnSampling,
            es: Array,
            spec: Spec,
            key: Array
    ):
        key_1, key_2 = jax.random.split(key)

        cme_1 = BootstrapConditionalTestEmbedding.from_data(
            kernel, dataset_1.xs, dataset_1.ys, dataset_1.xs, n_bootstrap=spec.n_bootstrap, key=key_1
        )

        cme_2 = BootstrapConditionalTestEmbedding.from_data(
            kernel, dataset_2.xs, dataset_2.ys, dataset_2.xs, n_bootstrap=spec.n_bootstrap, key=key_2
        )

        super().__init__(cme_1, cme_2, es)


def _empirical_cdf(distribution: np.ndarray, element: np.ndarray) -> np.ndarray:
    assert distribution.ndim == 1
    sorted_ratios = np.sort(distribution)
    left = np.searchsorted(sorted_ratios, element, side="left")
    right = np.searchsorted(sorted_ratios, element, side="right")
    return (right - left) / (2 * len(distribution))


class HuLeiTest(Test):
    test_statistic: Array

    class Spec(ABC):
        pass

    class SpecGroundTruthDensity(Spec):
        pass

    @dataclass(frozen=True)
    class SpecHomoscedasticGaussian(Spec):
        noise_std: float
        kernel: VectorKernelSpec
        p_train: float

        def __post_init__(self):
            assert self.noise_std > 0
            assert self.p_train >= 0
            assert self.p_train <= 1
            assert isinstance(self.kernel.y, LinearKernelSpec)

        def split(self, dataset: RKHSFnSampling, key: Array) -> tuple[RKHSFnSampling, RKHSFnSampling]:
            indices = jnp.arange(len(dataset))
            indices = jax.random.permutation(key, indices)
            n = int(len(dataset) * self.p_train)
            return dataset[indices[:n]], dataset[indices[n:]]

        def fit(self, dataset: RKHSFnSampling) -> CME:
            kernel = self.kernel.make()
            return kernel.cme(dataset.xs, dataset.ys)

    @staticmethod
    def gaussian_conditional_density_ratio(
            mean_1: Array, mean_2: Array, std_1: ArrayLike, std_2: ArrayLike, x: Array
    ) -> Array:
        return std_2 / std_1 * jnp.exp(0.5 * ((x - mean_2) / std_2) ** 2 - 0.5 * ((x - mean_1) / std_1) ** 2)

    @classmethod
    def from_homoscedastic_gaussian_model(
            cls,
            dataset_1: RKHSFnSampling, dataset_2: RKHSFnSampling,
            spec: SpecHomoscedasticGaussian,
            key: Array
    ):
        key_1, key_2 = jax.random.split(key)
        dataset_1_train, dataset_1_calibrate = spec.split(dataset_1, key_1)
        dataset_2_train, dataset_2_calibrate = spec.split(dataset_2, key_2)

        cme_1 = spec.fit(dataset_1_train)
        cme_2 = spec.fit(dataset_2_train)

        kmes_11 = cme_1(dataset_1_calibrate.xs)
        kmes_12 = cme_1(dataset_2_calibrate.xs)
        kmes_21 = cme_2(dataset_1_calibrate.xs)
        kmes_22 = cme_2(dataset_2_calibrate.xs)

        means_11 = jnp.einsum("...d,...d->...", kmes_11.coefficients, kmes_11.points)
        means_12 = jnp.einsum("...d,...d->...", kmes_12.coefficients, kmes_12.points)
        means_21 = jnp.einsum("...d,...d->...", kmes_21.coefficients, kmes_21.points)
        means_22 = jnp.einsum("...d,...d->...", kmes_22.coefficients, kmes_22.points)

        conditional_density_ratios_1 = cls.gaussian_conditional_density_ratio(
            mean_1=means_11, mean_2=means_21, std_1=spec.noise_std, std_2=spec.noise_std, x=dataset_1_calibrate.ys
        )

        conditional_density_ratios_2 = HuLeiTest.gaussian_conditional_density_ratio(
            mean_1=means_12, mean_2=means_22, std_1=spec.noise_std, std_2=spec.noise_std, x=dataset_2_calibrate.ys
        )

        return cls.from_ratios(conditional_density_ratios_1, conditional_density_ratios_2, key=key)

    @classmethod
    def from_ground_truth_density(
            cls,
            fn_1: RKHSFn, fn_2: RKHSFn,
            noise_1: GaussianNoiseSpec, noise_2: GaussianNoiseSpec,
            dataset_1: RKHSFnSampling, dataset_2: RKHSFnSampling,
            key: Array
    ) -> Array:
        assert noise_1.mean.size == 1
        assert noise_2.mean.size == 1
        assert noise_1.std.size == 1
        assert noise_2.std.size == 1

        noise_mean_1 = noise_1.mean.item()
        noise_mean_2 = noise_2.mean.item()
        noise_std_1 = noise_1.std.item()
        noise_std_2 = noise_2.std.item()

        conditional_density_ratios_1 = HuLeiTest.gaussian_conditional_density_ratio(
            mean_1=fn_1(dataset_1.xs) + noise_mean_1,
            mean_2=fn_2(dataset_1.xs) + noise_mean_2,
            std_1=noise_std_1,
            std_2=noise_std_2,
            x=dataset_1.ys
        )

        conditional_density_ratios_2 = HuLeiTest.gaussian_conditional_density_ratio(
            mean_1=fn_1(dataset_2.xs) + noise_mean_1,
            mean_2=fn_2(dataset_2.xs) + noise_mean_2,
            std_1=noise_std_1,
            std_2=noise_std_2,
            x=dataset_2.ys
        )

        return cls.from_ratios(conditional_density_ratios_1, conditional_density_ratios_2, key=key)

    @classmethod
    def from_ratios(
            cls,
            conditional_density_ratios_1: Array, conditional_density_ratios_2: Array,
            key: Array
    ):
        assert conditional_density_ratios_1.ndim == 1
        assert conditional_density_ratios_2.ndim == 1

        n_1 = len(conditional_density_ratios_1)
        n_2 = len(conditional_density_ratios_2)

        tie_break = jax.random.uniform(key, shape=(n_2,))
        equal = conditional_density_ratios_1[:, None] == conditional_density_ratios_2[None, :]
        less = conditional_density_ratios_1[:, None] < conditional_density_ratios_2[None, :]
        pairwise_comparison = less.astype(float) + equal * tie_break[None, :]

        cdf_values = 1 - _empirical_cdf(
            distribution=np.array(conditional_density_ratios_2),
            element=np.array(conditional_density_ratios_1)
        )

        mean_cdf_value = cdf_values.mean()
        var_cdf_value = jnp.sum((cdf_values - mean_cdf_value) ** 2) / (n_1 - 1)

        sigma = jnp.sqrt(var_cdf_value + n_1 / (12 * n_2))

        return (0.5 - pairwise_comparison.mean()) / (sigma / math.sqrt(n_1))

    def __init__(
            self,
            spec: Spec,
            fn_1: RKHSFn, fn_2: RKHSFn,
            noise_1: GaussianNoiseSpec, noise_2: GaussianNoiseSpec,
            dataset_1: RKHSFnSampling, dataset_2: RKHSFnSampling,
            key: Array
    ):
        if isinstance(spec, self.SpecGroundTruthDensity):
            self.test_statistic = self.from_ground_truth_density(
                fn_1=fn_1, fn_2=fn_2,
                noise_1=noise_1, noise_2=noise_2,
                dataset_1=dataset_1, dataset_2=dataset_2,
                key=key
            )
        elif isinstance(spec, self.SpecHomoscedasticGaussian):
            self.test_statistic = self.from_homoscedastic_gaussian_model(
                dataset_1=dataset_1, dataset_2=dataset_2,
                spec=spec,
                key=key
            )
        else:
            raise ValueError(f"Unknown HuLeiTest spec: {spec}")

    def _reject(self, significance_level: ArrayLike) -> Array:
        return jax.scipy.stats.norm.cdf(self.test_statistic) >= 1 - significance_level
