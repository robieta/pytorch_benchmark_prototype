from typing import List, Optional

import numpy as np


_IQR_WARN_THRESHOLD = 0.1
_IQR_GROSS_WARN_THRESHOLD = 0.25
_IQR_WARN_TEMPLATE = (
    "  WARNING: Interquartile range is {rel_iqr:.1f}% of the "
    "median measurement.\n"
    "           {guidance}"
)


def select_unit(t):
    time_unit = {-3: "ns", -2: "us", -1: "ms"}.get(
        int(np.log10(t) // 3), "s"
    )
    time_scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[time_unit]
    return time_unit, time_scale


class Measurement:
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
        self.env = env
        self.num_threads = num_threads
        self.stmt = stmt
        self.metadata = metadata

        self._sorted_times = sorted([t / number_per_run for t in times])
        self._median = np.median(self._sorted_times)
        self._bottom_quartile = np.percentile(self._sorted_times, 25)
        self._top_quartile = np.percentile(self._sorted_times, 75)
        self._iqr = self._top_quartile - self._bottom_quartile
        self._warnings = []

        rel_iqr = self._iqr / self._median * 100

        if self._iqr / self._median > _IQR_GROSS_WARN_THRESHOLD:
            self._warnings.append(_IQR_WARN_TEMPLATE.format(
                rel_iqr=rel_iqr,
                guidance="This is well outside of expected bounds and "
                "suggests significant environmental influence."))
        elif self._iqr / self._median > _IQR_WARN_THRESHOLD:
            self._warnings.append(_IQR_WARN_TEMPLATE.format(
                rel_iqr=rel_iqr,
                guidance="This could indicate system fluctuation."))

    def __getstate__(self):
        return {
            "label": self.label,
            "sub_label": self.sub_label,
            "description": self.description,
            "env": self.env,
            "num_threads": self.num_threads,
            "number_per_run": self.number_per_run,
            "times": self.times,
            "stmt": self.stmt,
            "metadata": self.metadata,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    @property
    def mean(self):
        return np.mean(self._sorted_times)

    @property
    def median(self):
        return self._median

    @property
    def title(self):
        if self.label is not None:
            label = self.label
        elif isinstance(self.stmt, str):
            label = self.stmt
        else:
            label = "[Missing primary label]"

        return label + (f": {self.sub_label}" if self.sub_label else "")

    @property
    def as_row_name(self):
        return self.sub_label or self.stmt or "[Unknown]"

    def __repr__(self):
        return super().__repr__()
        # repr = [super().__repr__(), "\n", self.title, "\n"]

        # time_unit, time_scale = select_unit(self.median)
        # repr.extend(
        #     [
        #         f"  Median: {self._median / time_scale:.2f} {time_unit}\n",
        #         f"  IQR:    {self._iqr / time_scale:.2f} {time_unit} "
        #         f"({self._bottom_quartile / time_scale:.2f} to {self._top_quartile / time_scale:.2f})\n",
        #     ]
        # )
        # repr.extend([
        #     f"  {len(self.times)} measurements, "
        #     f"{self.number_per_run} runs per measurement, "
        #     f"{self.num_threads} thread{'s' if self.num_threads > 1 else ''}\n"
        # ])
        # repr.extend(self._warnings)

        # return "".join(repr).strip()
