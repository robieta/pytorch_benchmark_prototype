"""Trivial use of revised Timer API:

$ python -m examples.simple_timeit
"""

import torch

import utils as benchmark_utils

def main():
    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals={"x": torch.ones((4, 8)), "y": torch.ones((1, 8))},
        label="Broadcasting add (4x8)",
    )

    for _ in range(3):
        print(timer.blocked_autorange(), "\n")


if __name__ == "__main__":
    main()
