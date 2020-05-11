"""Timer class based on the timeit.Timer class, but torch aware."""

import logging
import sys
import timeit
from typing import Callable, Optional

import numpy as np
import torch

from . import common


if torch.has_cuda:
    def timer():
        torch.cuda.synchronize()
        return timeit.default_timer()
else:
    timer = timeit.default_timer


_MAX_RERUN_ON_WARNINGS = 5


class Timer(object):
    def __init__(
        self,
        stmt="pass",
        setup="pass",
        timer=timer,
        globals: Optional[dict] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        env: Optional[str] = None,
        num_threads=1,
    ):
        if not isinstance(stmt, str):
            raise ValueError("Currently only a `str` stmt is supported.")
        self._stmt = stmt
        self._setup = setup
        self._timer = timer
        self._globals = globals
        self._gen_globals = isinstance(globals, common.ExampleGenerator)

        self._label = label
        self._sub_label = sub_label
        if self._gen_globals and description is not None:
            raise ValueError(
                "`description` should not be provided when globals is an "
                "ExampleGenerator. Include it in the `description` field "
                "of the Examples instead."
            )
        self._description = description
        self._env = env
        self._num_threads = num_threads

        # Make sure the init args are valid.
        g = next(globals.take_internal(1)).globals if self._gen_globals else globals
        t = timeit.Timer(stmt=stmt, setup=setup, timer=timer, globals=g)

        self.t = None if self._gen_globals else t

    def _timer_iter(self, n):
        if n is not None and not self._gen_globals:
            raise ValueError(
                "`n` should only be specified if `globals` is an "
                "ExampleGenerator."
            )
        if self._gen_globals and n is None:
            raise ValueError(
                "`n` must be specified if `globals` "
                "is an ExampleGenerator."
            )

        if self._gen_globals:
            for example in self._globals.take_internal(n):
                timer = timeit.Timer(
                    stmt=self._stmt,
                    setup=self._setup,
                    timer=self._timer,
                    globals=example.globals,
                )
                yield timer, example
        else:
            example = common.Example(
                globals=self._globals,
                description=self._description,
                metadata=None
            )
            yield self.t, example

    def _blocked_autorange(
        self,
        timer,  # type annotating causes a crash: https://bugs.python.org/issue40595
        callback: Optional[Callable],
        min_run_time: float,
        description: Optional[str] = None,
        metadata: Optional[str] = None,
    ):
        # Estimate the block size needed for measurement to be negligible
        # compared to the inner loop. This also serves as a warmup.
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

        return common.Measurement(
            number_per_run=number,
            times=times,
            num_threads=self._num_threads,
            label=self._label,
            sub_label=self._sub_label,
            description=description,
            env=self._env,
            stmt=self._stmt,
            metadata=metadata,
        )

    def timeit(self, number=1000000, n_trials=None):
        output = []
        with set_torch_threads(self._num_threads):
            for i, (timer, example) in enumerate(self._timer_iter(n_trials)):
                # Warmup
                timer.timeit(number=max(int(number // 100, 1)))

                outout.append(common.Measurement(
                    number_per_run=number,
                    times=[timer.timeit(number=number)],
                    num_threads=self._num_threads,
                    label=self._label,
                    sub_label=self._sub_label,
                    description=example.description,
                    env=self._env,
                    stmt=self._stmt,
                    metadata=example.metadata,
                ))
        return output if self._gen_globals else output[0]

    def repeat(self, repeat=-1, number=-1):
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def autorange(self, callback=None):
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def blocked_autorange(
        self,
        callback: Optional[Callable] = None,
        min_run_time: float = 0.2,
        rerun_on_warning: bool = False,
        n: Optional[int] = None,
        display_progress: bool = False,
    ):
        output = []
        def collect_measurement(timer, description=None, metadata=None):
            measure = lambda: self._blocked_autorange(
                timer=timer,
                callback=callback,
                min_run_time=min_run_time,
                description=description,
                metadata=metadata,
            )

            measurement = measure()
            count = 1
            while rerun_on_warning and measurement._warnings:
                if count == _MAX_RERUN_ON_WARNINGS:
                    logging.warning(
                        f" Trial still has warnings after {count} attempts. "
                        + f"Aborting reruns. {measurement.title}"
                    )
                    break
                measurement = measure()
                count += 1

            output.append(measurement)

        with set_torch_threads(self._num_threads):
            for i, (timer, example) in enumerate(self._timer_iter(n)):
                collect_measurement(timer, example.description, example.metadata)
                if display_progress:
                    print(f"\r{i + 1} / {n} ", end="")
                    sys.stdout.flush()
        if display_progress:
            print()

        return output if self._gen_globals else output[0]
