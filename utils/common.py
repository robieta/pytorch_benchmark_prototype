"""Base shared classes and utilities."""

from typing import List, NamedTuple, Optional

import numpy as np


_MAX_SIGNIFICANT_FIGURES = 4
_MIN_CONFIDENCE_INTERVAL = 25e-9  # 25 ns

# Measurement will include a warning if the distribution is suspect. All
# runs are expected to ahave some variation; these parameters set the
# thresholds.
_IQR_WARN_THRESHOLD = 0.1
_IQR_GROSS_WARN_THRESHOLD = 0.25


def select_unit(t: float):
    """Determine how to scale times for O(1) magnitude.

    This utility is used to format numbers for human consumption.
    """
    time_unit = {-3: "ns", -2: "us", -1: "ms"}.get(int(np.log10(t) // 3), "s")
    time_scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[time_unit]
    return time_unit, time_scale


def unit_to_english(u: str) -> str:
    return {
        "ns": "nanosecond",
        "us": "microsecond",
        "ms": "millisecond",
        "s": "second",
    }[u]


def trim_sigfig(x: float, n: int) -> float:
    """Trim `x` to `n` significant figures. (e.g. 3.14159, 2 -> 3.10000)"""
    assert n == int(n)
    magnitude = int(np.ceil(np.log10(np.abs(x))))
    scale = 10 ** (magnitude - n)
    return np.round(x / scale) * scale


class Measurement:
    """The result of a Timer measurement.

    This class stores one or more measurements of a given statement. It is
    serializable and provides several convenience methods
    (including a detailed __repr__) for downstream consumers.
    """
    def __init__(
        self,
        number_per_run: int,
        times: List[float],
        num_threads: int,
        label: Optional[str],
        sub_label: Optional[str],
        description: Optional[str],
        env: Optional[str],
        stmt: Optional[str],
        metadata: Optional[dict],
    ):
        self.number_per_run = number_per_run
        self.times = times
        self.label = label
        self.sub_label = sub_label
        self.description = description
        self._env = env
        self.num_threads = num_threads
        self.stmt = stmt
        self.metadata = metadata

        # Derived attributes
        self._sorted_times = sorted([t / number_per_run for t in times])
        self._median = np.median(self._sorted_times)
        self._bottom_quartile = np.percentile(self._sorted_times, 25)
        self._top_quartile = np.percentile(self._sorted_times, 75)
        self._iqr = self._top_quartile - self._bottom_quartile
        self._warnings = self._populate_warnings()

    # Pickle support.
    def __getstate__(self):
        return {
            "label": self.label,
            "sub_label": self.sub_label,
            "description": self.description,
            "env": self._env,
            "num_threads": self.num_threads,
            "number_per_run": self.number_per_run,
            "times": self.times,
            "stmt": self.stmt,
            "metadata": self.metadata,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def _populate_warnings(self):
        warnings, rel_iqr = [], self._iqr / self._median * 100
        add_warning = lambda msg: warnings.append(
            f"  WARNING: Interquartile range is {rel_iqr:.1f}% "
            f"of the median measurement.\n           {msg}"
        )

        if self._iqr / self._median > _IQR_GROSS_WARN_THRESHOLD:
            add_warning("This suggests significant environmental influence.")
        elif self._iqr / self._median > _IQR_WARN_THRESHOLD:
            add_warning("This could indicate system fluctuation.")
        return warnings

    @property
    def median(self) -> float:
        return self._median

    @property
    def significant_figures(self) -> int:
        """Approximate significant figure estimate.

        This property is intended to give a convenient way to estimate the
        precision of a measurement. It only uses the interquartile region to
        estimate statistics to try to mitigate skew from the tails, and
        uses a static z value of 1.645 since it is not expected to be used
        for small values of `n`, so z can approximate `t`.

        The significant figure estimation is uses in conjunction with the
        `trim_sigfig` method to provide a more human interpretable data
        summary. __repr__ does not use this method; it simply displays raw
        values. Significant figure estimation is intended for `Compare`.
        """
        n_total = len(self._sorted_times)
        lower_bound = int(n_total // 4)
        upper_bound = int(np.ceil(3 * n_total / 4))
        interquartile_points = self._sorted_times[lower_bound:upper_bound]
        std = np.std(interquartile_points)
        sqrt_n = np.sqrt(len(interquartile_points))

        # Rough estimates. These are by no means statistically rigourous.
        confidence_interval = max(1.645 * std / sqrt_n, _MIN_CONFIDENCE_INTERVAL)
        relative_ci = np.log10(self._median / confidence_interval)
        num_significant_figures = int(np.floor(relative_ci))
        return min(max(num_significant_figures, 1), _MAX_SIGNIFICANT_FIGURES)

    @property
    def title(self) -> str:
        """Best effort attempt at a string label for the measurement."""
        if self.label is not None:
            label = self.label
        elif isinstance(self.stmt, str):
            label = self.stmt
        else:
            label = "[Missing primary label]"

        return label + (f": {self.sub_label}" if self.sub_label else "")

    @property
    def env(self) -> str:
        return "Unspecified env" if self._env is None else self._env

    @property
    def as_row_name(self) -> str:
        return self.sub_label or self.stmt or "[Unknown]"

    def __repr__(self):
        """
        Example repr:
            <utils.common.Measurement object at 0x7f395b6ac110>
              Broadcasting add (4x8)
              Median: 5.73 us
              IQR:    2.25 us (4.01 to 6.26)
              372 measurements, 100 runs per measurement, 1 thread
              WARNING: Interquartile range is 39.4% of the median measurement.
                       This suggests significant environmental influence.
        """
        repr = [super().__repr__(), "\n", self.title, "\n"]

        time_unit, time_scale = select_unit(self.median)
        repr.extend(
            [
                f"  Median: {self._median / time_scale:.2f} {time_unit}\n",
                f"  IQR:    {self._iqr / time_scale:.2f} {time_unit} "
                f"({self._bottom_quartile / time_scale:.2f} to {self._top_quartile / time_scale:.2f})\n",
            ]
        )
        repr.extend(
            [
                f"  {len(self.times)} measurements, "
                f"{self.number_per_run} runs per measurement, "
                f"{self.num_threads} thread{'s' if self.num_threads > 1 else ''}\n"
            ]
        )
        repr.extend(self._warnings)

        return "".join(repr).strip()


class Example(NamedTuple):
    globals: dict
    description: Optional[str]
    metadata: Optional[dict]


class ExampleGenerator(object):
    def take(self, n: int) -> Example:
        """Subclasses should override this method to yield Example instances."""
        raise NotImplementedError

    def take_internal(self, n: int) -> Example:
        """Helper method to check user provides examples"""
        for i, example in enumerate(self.take(n)):
            if not isinstance(example, Example):
                raise ValueError(
                    "`.take` should yield Examples," f"got {type(i)} instead"
                )
            yield example

        if (i + 1) != n:
            logging.warning(
                f" Expected {n} examples, but {i + 1} were " "produced by `.take`"
            )
