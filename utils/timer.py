import inspect
import logging
import timeit
import typing

import numpy as np
import torch

if torch.has_cuda:
    def timer():
        torch.cuda.synchronize()
        return timeit.default_timer()
else:
    timer = timeit.default_timer


_IQR_WARN_THRESHOLD = 0.05
_MAX_RERUN_ON_WARNINGS = 5


class Measurement:
    def __init__(self, label: typing.Optional[str],
                 sub_label: typing.Optional[str], num_threads: int,
                 number_per_run: int, times: typing.List[float],
                 stmt: typing.Optional[str],
                 metadata: typing.Optional[dict]):
        self.label = label
        self.sub_label = sub_label
        self.num_threads = num_threads
        self.number_per_run = number_per_run
        self.times = times
        self.stmt = stmt
        self.metadata = metadata

        self._sorted_times = sorted([t / number_per_run for t in times])
        self._median = np.median(self._sorted_times)
        self._bottom_quartile = np.percentile(self._sorted_times, 25)
        self._top_quartile = np.percentile(self._sorted_times, 75)
        self._iqr = self._top_quartile - self._bottom_quartile
        self._warnings = []
        if self._iqr / self._median > _IQR_WARN_THRESHOLD:
            rel_iqr = self._iqr / self._median * 100
            self._warnings.append(
                f"  WARNING: Interquartile range is {rel_iqr:.1f}% of the "
                "median measurement.\n"
                f"{' ' * 11}This could indicate system fluctuation.\n",
            )

    def __getstate__(self):
        return {
            "label": self.label, "sub_label": self.sub_label,
            "num_threads": self.num_threads,
            "number_per_run": self.number_per_run, "times": self.times,
            "stmt": self.stmt, "metadata": metadata
        }

    def __setstate__(self, state):
        self.__init__(**state)

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
    def mean(self):
        return np.mean(self._sorted_times)

    @property
    def median(self):
        return self._median

    def __repr__(self):
        repr = [super().__repr__(), "\n", self.title, "\n"]

        time_unit = ({-3: "ns", -2: "us", -1: "ms"}
            .get(int(np.log10(self._median) // 3), "s"))
        time_scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[time_unit]

        repr.extend([
            f"  Median: {self._median / time_scale:.2f} {time_unit}\n",
            f"  IQR:    {self._iqr / time_scale:.2f} {time_unit} "
            f"({self._bottom_quartile / time_scale:.2f} to {self._top_quartile / time_scale:.2f})\n"
        ])
        repr.append(f"  {len(self.times)} measurements, {self.number_per_run} runs per measurement\n")
        repr.append(f"  {self.num_threads} thread{'s' if self.num_threads > 1 else ''}\n")
        repr.extend(self._warnings)

        return "".join(repr).strip()


class Example(typing.NamedTuple):
    globals: dict
    sub_label: typing.Optional[str]
    metadata: typing.Optional[dict]


class ExampleGenerator(object):
    default_number = 10
    def take(self, n):
        raise NotImplementedError

    def take_internal(self, n):
        if n is None:
            n = self.default_number
        for i in self.take(n):
            assert isinstance(i, Example)
            yield i


class Timer(object):
    def __init__(self, stmt="pass", setup="pass", timer=timer, globals=None,
                 label=None, num_threads=1):
        self._stmt = stmt
        self._setup = setup
        self._timer = timer
        self._globals = globals
        self._gen_globals = isinstance(globals, ExampleGenerator)

        self._label = label
        self._num_threads = num_threads

        # Make sure the init args are valid.
        t = timeit.Timer(
            stmt=stmt, setup=setup, timer=timer,
            globals=next(globals.take_internal(1)).globals if self._gen_globals
                    else globals)

        self.t = None if self._gen_globals else t

    def autorange(self, callback=None):
        raise NotImplementedError

    def _blocked_autorange(self, timer: timeit.Timer, callback, min_run_time, sub_label=None, metadata=None):
        overhead = np.median([timer.timeit(0) for _ in range(5)])
        number = 1
        while True:
            time_taken = timer.timeit(number)
            relative_overhead = overhead / time_taken
            if overhead <= 1e-5 and time_taken >= min_run_time / 1000:
                break
            number *= 10

        total_time = 0
        times = []
        while total_time < min_run_time:
            time_taken = timer.timeit(number)
            total_time += time_taken
            if callback:
                callback(number, time_taken)
            times.append(time_taken)

        return Measurement(
            label=self._label, sub_label=sub_label,
            num_threads=self._num_threads, number_per_run=number, times=times,
            stmt=self._stmt, metadata=metadata)

    def blocked_autorange(self, callback=None, min_run_time=0.2, rerun_on_warning=False, n=None):
        if n is not None and not self._gen_globals:
            raise ValueError("`n` should only be specified if `globals` is an "
                             "ExampleGenerator. (e.g. a Fuzzer)")
        output = []
        prior_num_threads = torch.get_num_threads()
        torch.set_num_threads(self._num_threads)

        def collect_measurement(timer, sub_label=None, metadata=None):
            measure = lambda: self._blocked_autorange(
                timer=timer, callback=callback, min_run_time=min_run_time,
                sub_label=sub_label, metadata=metadata)

            measurement = measure()
            count = 1
            while rerun_on_warning and measurement._warnings:
                if count == _MAX_RERUN_ON_WARNINGS:
                    logging.warning(f" Trial still has warnings after {count} attempts. " +
                                    f"Aborting reruns. {measurement.title}")
                    break
                measurement = measure()
                count += 1

            return measurement

        if self._gen_globals:
            for example in self._globals.take_internal(n):
                timer = timeit.Timer(
                    stmt=self._stmt, setup=self._setup, timer=self._timer,
                    globals=example.globals)
                output.append(collect_measurement(timer, example.sub_label, metadata=example.metadata))
        else:
            output.append(collect_measurement(self.t))

        torch.set_num_threads(prior_num_threads)
        return output
