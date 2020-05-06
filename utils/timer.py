import inspect
import logging
import timeit
from typing import Callable, NamedTuple, Optional

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


class Example(NamedTuple):
    globals: dict
    description: Optional[str]
    metadata: Optional[dict]


class ExampleGenerator(object):
    def take(self, n: int):
        raise NotImplementedError

    def take_internal(self, n: int):
        for i, example in enumerate(self.take(n)):
            if not isinstance(example, Example):
                raise ValueError("`.take` should yield Examples,"
                                 f"got {type(i)} instead")
            yield example

        if (i + 1) != n:
            logging.warning(f" Expected {n} examples, but {i + 1} were "
                            "produced by `.take`")


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
        self._stmt = stmt
        self._setup = setup
        self._timer = timer
        self._globals = globals
        self._gen_globals = isinstance(globals, ExampleGenerator)

        self._label = label
        self._sub_label = sub_label
        if self._gen_globals and description is not None:
            raise ValueError(
                "`description` should not be provided when globals is an "
                "ExampleGenerator. Include it in the `description` field "
                "of the Examples instead.")
        self._description = description
        self._env = env
        self._num_threads = num_threads

        # Make sure the init args are valid.
        g = next(globals.take_internal(1)).globals if self._gen_globals else globals
        t = timeit.Timer(
            stmt=stmt,
            setup=setup,
            timer=timer,
            globals=g,
        )

        self.t = None if self._gen_globals else t

    #TODO(robieta): `def timeit(self):`

    def autorange(self, callback=None):
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def _blocked_autorange(
        self,
        timer: timeit.Timer,
        callback: Optional[Callable],
        min_run_time: float,
        description: Optional[str]=None,
        metadata: Optional[str]=None
    ):
        # Estimate the block size needed for measurement to be negligible
        # compared to the inner loop.
        overhead = np.median([timer.timeit(0) for _ in range(5)])
        number = 1
        while True:
            time_taken = timer.timeit(number)
            relative_overhead = overhead / time_taken
            if overhead <= 1e-5 and time_taken >= min_run_time / 1000:
                break
            number *= 10

        # Don't waste the last measurement of the block size determination.
        total_time = time_taken
        times = [time_taken]

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


    def blocked_autorange(
        self,
        callback: Optional[Callable]=None,
        min_run_time: float=0.2,
        rerun_on_warning: bool=False,
        n: Optional[int]=None
    ):
        if n is not None and not self._gen_globals:
            raise ValueError(
                "`n` should only be specified if `globals` is an "
                "ExampleGenerator.")
        if self._gen_globals and n is None:
            raise ValueError("`n` must be specified if `globals` "
                             "is an ExampleGenerator.")

        prior_num_threads = torch.get_num_threads()
        torch.set_num_threads(self._num_threads)

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

            return measurement

        if self._gen_globals:
            output = []
            for example in self._globals.take_internal(n):
                timer = timeit.Timer(
                    stmt=self._stmt,
                    setup=self._setup,
                    timer=self._timer,
                    globals=example.globals,
                )
                output.append(
                    collect_measurement(
                        timer, example.description, example.metadata
                    )
                )
        else:
            output = collect_measurement(self.t, self._description)

        torch.set_num_threads(prior_num_threads)
        return output
