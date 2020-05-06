import itertools as it
import pickle
import sys
import time

import torch

import utils as benchmark_utils

def simple_broadcasting_add():
    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals={"x": torch.ones((4, 8)), "y": torch.ones((1, 8))},
        label="Broadcasting add (4x8)",
    )

    for _ in range(3):
        print(timer.blocked_autorange(), "\n")


def example_gen_broadcasting_add():
    class MyExampleGenerator(benchmark_utils.ExampleGenerator):
        default_number = 5
        def take(self, n):
            for i in range(n):
                size = (i + 1) ** 6
                yield benchmark_utils.Example(
                    globals={"x": torch.ones((size, 8)), "y": torch.ones((1, 8))},
                    description=f"({size}, 8) + (1, 8)",
                    metadata={"size": size, "num_elements": size * 8})

    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals=MyExampleGenerator(),
        label="Broadcasting add",
    )
    for i in timer.blocked_autorange(n=3):
        print(i)
        print(f"{i.median / i.metadata['num_elements'] * 1e9:.1f} ns / element\n")


def compare():
    class FauxTorch(object):
        """Emulate different versions of pytorch.

        In normal circumstances this would be done with multiple processes
        writing serialized measurements, but this simplifies that model to
        make the example clearer.
        """
        def __init__(self, real_torch, extra_ns_per_element):
            self._real_torch = real_torch
            self._extra_ns_per_element = extra_ns_per_element

        def extra_overhead(self, result):
            time.sleep(result.numel() * self._extra_ns_per_element * 1e-9)
            return result

        def add(self, *args, **kwargs):
            return self.extra_overhead(self._real_torch.add(*args, **kwargs))

        def mul(self, *args, **kwargs):
            return self.extra_overhead(self._real_torch.mul(*args, **kwargs))

        def cat(self, *args, **kwargs):
            return self.extra_overhead(self._real_torch.cat(*args, **kwargs))

        def matmul(self, *args, **kwargs):
            return self.extra_overhead(self._real_torch.matmul(*args, **kwargs))

    tasks = {
        "add": "torch.add(x, y)",
        # "mul": "torch.mul(x, y)",
        # "cat": "torch.cat((x, y), dim=0)",
        # "matmul": "torch.matmul(x, y.transpose(0, 1))",
    }

    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "torch": FauxTorch(torch, overhead_ns),
                "x": torch.ones((size, 4)),
                "y": torch.ones((1, 4)),
            },
            label=label,
            description=f"({size}, 4), (1, 4)",
            env=branch,
            num_threads=num_threads,
        )
        for branch, overhead_ns in [("master", 10), ("my_super_fast_branch", 5)]
        for label, stmt in tasks.items()
        for size in [1, 1000, 10000]
        for num_threads in [1, 4]
    ]

    for i, timer in enumerate(timers * repeats):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])
    comparison.print()



if __name__ == "__main__":
    # simple_broadcasting_add()
    # example_gen_broadcasting_add()
    compare()
