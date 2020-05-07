"""Trivial use of the Fuzzer API:

$ python -m examples.simple_fuzzer
"""

import numpy as np
import torch

import utils as benchmark_utils


def main():
    class AddFuzzer(benchmark_utils.Fuzzer):
        """Add custom description and expand metadata.

        This class is mostly string munging to make a well formatted
        description.
        """
        def take(self, n):
            for example in super(AddFuzzer, self).take(n):
                x, y = example.globals["x"], example.globals["y"]
                description = (
                    f"{x.numel():>7}" + " | " +
                    f"{', '.join(tuple(f'{i:>4}' for i in x.shape))}".ljust(16) + " | " +
                    (f"{example.metadata['x_order']}" if not x.is_contiguous()
                     else "contiguous").ljust(15) + " | " +
                    (f"{example.metadata['y_order']}" if not y.is_contiguous()
                     else "contiguous").ljust(15))
                example.metadata["numel"] = int(x.numel())
                yield benchmark_utils.Example(
                    example.globals, description, example.metadata)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # == This is the Fuzzer API. ==============================================
    # /////////////////////////////////////////////////////////////////////////
    add_fuzzer = AddFuzzer(
        parameters=[
            benchmark_utils.FuzzedParameter("k0", 16, 16 * 1024, "loguniform"),
            benchmark_utils.FuzzedParameter("k1", 16, 16 * 1024, "loguniform"),
            benchmark_utils.FuzzedParameter("k2", 16, 16 * 1024, "loguniform"),
            benchmark_utils.FuzzedParameter("d", distribution={2: 0.6, 3: 0.4}),
        ],
        tensors=[
            benchmark_utils.FuzzedTensor(
                name="x", size=("k0", "k1", "k2"), dim_parameter="d",
                probability_contiguous=0.75, min_elements=64 * 1024,
                max_elements=128 * 1024,
            ),
            benchmark_utils.FuzzedTensor(
                name="y", size=("k0", "k1", "k2"), dim_parameter="d",
                probability_contiguous=0.75, min_elements=64 * 1024,
                max_elements=128 * 1024,
            ),
        ],
        seed=0,
    )

    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals=add_fuzzer,
    )

    measurements = timer.blocked_autorange(
        min_run_time=0.1, n=250,
        display_progress=True,
        rerun_on_warning=True)

    # More string munging to make pretty output.
    print(f"Average attemts per valid config: {1. / (1. - add_fuzzer.rejection_rate):.1f}")
    time_fn = lambda m: m.median / m.metadata["numel"]
    measurements.sort(key=time_fn)

    template = f"{{:>6}}{' ' * 19}Size    Shape{' ' * 13}X order           Y order\n{'-' * 80}"
    print(template.format("Best:"))
    for m in measurements[:15]:
        print(f"{time_fn(m) * 1e9:>4.1f} ns / element     {m.description}")

    print("\n" + template.format("Worst:"))
    for m in measurements[-15:]:
        print(f"{time_fn(m) * 1e9:>4.1f} ns / element     {m.description}")


if __name__ == "__main__":
    main()
