#!/usr/bin/env python3
"""Cross-immunity kernels, and the saturation property they exist to provide.

The central claim in the script's docstring is that escaping a *nearby* strain
is worth far more than moving the same distance away from a distant one. That is
a property of the kernel, so it is tested as one -- with the exact numbers quoted
in the README.
"""

from __future__ import annotations

import numpy as np
import pytest

import plant_population_escape as pop

pytestmark = pytest.mark.unit

SCALE = 2.0


def escape_at(distance, kernel="exponential", scale=SCALE):
    """Escape from one strain at ``distance``: 1 - cross-immunity."""
    return 1.0 - float(pop.cross_immunity(np.array([float(distance)]), scale, kernel)[0])


class TestExponentialKernel:
    def test_full_protection_at_zero_distance(self):
        assert pop.cross_immunity(np.array([0.0]), SCALE, "exponential")[0] == pytest.approx(1.0)

    def test_one_over_e_at_the_scale(self):
        value = pop.cross_immunity(np.array([SCALE]), SCALE, "exponential")[0]
        assert value == pytest.approx(1.0 / np.e)

    def test_monotone_decreasing(self):
        distances = np.linspace(0.0, 20.0, 200)
        values = pop.cross_immunity(distances, SCALE, "exponential")
        assert np.all(np.diff(values) < 0)

    def test_bounded_in_zero_one(self):
        distances = np.linspace(0.0, 100.0, 500)
        values = pop.cross_immunity(distances, SCALE, "exponential")
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_never_quite_reaches_zero(self):
        assert pop.cross_immunity(np.array([1e6]), SCALE, "exponential")[0] >= 0.0


class TestTheLocalImpactProperty:
    """The README's headline table, asserted rather than asserted-in-prose."""

    def test_readme_numbers_for_a_near_move(self):
        assert escape_at(1.0) == pytest.approx(0.393, abs=5e-4)
        assert escape_at(1.5) == pytest.approx(0.528, abs=5e-4)

    def test_readme_numbers_for_a_far_move(self):
        assert escape_at(10.0) == pytest.approx(0.993, abs=5e-4)
        assert escape_at(11.0) == pytest.approx(0.996, abs=5e-4)

    def test_half_a_unit_near_beats_a_whole_unit_far(self):
        near_gain = escape_at(1.5) - escape_at(1.0)
        far_gain = escape_at(11.0) - escape_at(10.0)
        assert near_gain > far_gain
        assert near_gain / far_gain > 20.0

    def test_the_gain_from_a_fixed_step_decays_monotonically(self):
        """Same 0.5-unit step, taken from further and further out."""
        gains = [escape_at(start + 0.5) - escape_at(start) for start in np.arange(0.0, 10.0, 0.5)]
        assert gains == sorted(gains, reverse=True)

    def test_the_gradient_is_steepest_at_zero(self):
        step = 1e-6
        gradients = [
            (escape_at(d + step) - escape_at(d)) / step for d in (0.0, 1.0, 5.0, 10.0)
        ]
        assert gradients[0] == max(gradients)
        assert gradients == sorted(gradients, reverse=True)


class TestSigmoidKernel:
    def test_half_protection_at_the_scale(self):
        value = pop.cross_immunity(np.array([SCALE]), SCALE, "sigmoid")[0]
        assert value == pytest.approx(0.5)

    def test_plateaus_near_full_protection_at_short_range(self):
        assert pop.cross_immunity(np.array([0.0]), SCALE, "sigmoid")[0] > 0.98

    def test_monotone_decreasing(self):
        values = pop.cross_immunity(np.linspace(0.0, 20.0, 200), SCALE, "sigmoid")
        assert np.all(np.diff(values) < 0)

    def test_it_flattens_the_local_gradient_as_documented(self):
        """The documented trade-off: a sigmoid is more realistic but blunts
        exactly the near-field sensitivity this analysis measures."""
        step = 1e-6
        exponential = (escape_at(0.0 + step) - escape_at(0.0)) / step
        sigmoid = (
            escape_at(0.0 + step, "sigmoid") - escape_at(0.0, "sigmoid")
        ) / step
        assert sigmoid < exponential


class TestLinearKernel:
    def test_full_protection_at_zero(self):
        assert pop.cross_immunity(np.array([0.0]), SCALE, "linear")[0] == pytest.approx(1.0)

    def test_zero_at_and_beyond_the_scale(self):
        beyond = pop.cross_immunity(np.array([SCALE, SCALE * 2, 100.0]), SCALE, "linear")
        assert beyond == pytest.approx([0.0, 0.0, 0.0])

    def test_linear_in_between(self):
        assert pop.cross_immunity(np.array([SCALE / 2]), SCALE, "linear")[0] == pytest.approx(0.5)

    def test_clipped_not_negative(self):
        assert pop.cross_immunity(np.array([1e6]), SCALE, "linear")[0] == 0.0

    def test_constant_gain_makes_it_the_wrong_default(self):
        """A linear kernel gives the same value to every unit inside the scale,
        which is the behaviour the exponential exists to avoid."""
        first = escape_at(0.5, "linear") - escape_at(0.0, "linear")
        second = escape_at(1.5, "linear") - escape_at(1.0, "linear")
        assert first == pytest.approx(second)


class TestKernelValidation:
    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            pop.cross_immunity(np.array([1.0]), SCALE, "gaussian")

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    @pytest.mark.parametrize("kernel", ["exponential", "sigmoid", "linear"])
    def test_non_positive_scale_raises(self, bad, kernel):
        with pytest.raises(ValueError, match="cross-immunity-scale"):
            pop.cross_immunity(np.array([1.0]), bad, kernel)

    def test_shape_is_preserved(self):
        distances = np.zeros((4, 7))
        assert pop.cross_immunity(distances, SCALE, "exponential").shape == (4, 7)


class TestScaleSensitivity:
    def test_a_larger_scale_means_more_cross_protection(self):
        near = pop.cross_immunity(np.array([2.0]), 1.0, "exponential")[0]
        far = pop.cross_immunity(np.array([2.0]), 10.0, "exponential")[0]
        assert far > near

    def test_escape_is_scale_free_in_the_ratio(self):
        """Doubling both the distance and the scale leaves cross-immunity fixed."""
        assert pop.cross_immunity(np.array([3.0]), 2.0, "exponential")[0] == pytest.approx(
            pop.cross_immunity(np.array([6.0]), 4.0, "exponential")[0]
        )
