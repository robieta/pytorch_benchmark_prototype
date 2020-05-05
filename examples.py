import torch

import utils as benchmark_utils

def simple_broadcasting_add():
    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals={"x": torch.ones((4, 8)), "y": torch.ones((1, 8))},
        label="Broadcasting add (4x8)",
    )

    for _ in range(3):
        print(timer.blocked_autorange()[0], "\n")

def example_gen_broadcasting_add():
    class MyExampleGenerator(benchmark_utils.ExampleGenerator):
        default_number = 5
        def take(self, n):
            for i in range(n):
                size = (i + 1) ** 6
                yield benchmark_utils.Example(
                    globals={"x": torch.ones((size, 8)), "y": torch.ones((1, 8))},
                    sub_label=f"({size}, 8) + (1, 8)",
                    metadata={"size": size, "num_elements": size * 8})

    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals=MyExampleGenerator(),
        label="Broadcasting add",
    )
    for i in timer.blocked_autorange():
        print(i)
        print(f"{i.median / i.metadata['num_elements'] * 1e9:.1f} ns / element\n")


if __name__ == "__main__":
    simple_broadcasting_add()
    # example_gen_broadcasting_add()
