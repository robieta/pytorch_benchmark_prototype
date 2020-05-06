"""Trivial use of revised Timer API:

$ python -m examples.intermediate_timeit
"""

import torch

import utils as benchmark_utils

def main():
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


if __name__ == "__main__":
    main()
