"""End-to-end example which uses all three major APIs:

$ python -m examples.advanced

NOTE:
  This example assumes that you have the following environments
  built from the following branches:
    master: master
    branch: gh/taylorrobie/unify_index
"""

import argparse
import os
import pickle
import sys
import tempfile
import subprocess

import numpy as np
import torch

import utils as benchmark_utils


_NUM_RUNS = 20
_REPEATS = 2
_ENVS = ("master", "branch")
_CONFIGURATIONS = (
    ("master", "N/A"),
    ("branch", "auto"),
    ("branch", "batch_major"),
    ("branch", "feature_major"),
)


class GatherFuzzer(benchmark_utils.Fuzzer):
    def __init__(self):
        super(GatherFuzzer, self).__init__(
            parameters = [
                benchmark_utils.FuzzedParameter("k0", 4, 16 * 1024, "loguniform"),
                benchmark_utils.FuzzedParameter("k1", 4, 16 * 1024, "loguniform"),
                benchmark_utils.FuzzedParameter("k2", 4, 16 * 1024, "loguniform"),
                benchmark_utils.FuzzedParameter("m", 4, 100000, "loguniform"),
                benchmark_utils.FuzzedParameter("d", distribution={2: 0.6, 3: 0.4}),
                benchmark_utils.FuzzedParameter("dim", 0, 2, "uniform", constraint=lambda d, dim, **kwargs: dim < d ),
            ],
            tensors = [
                benchmark_utils.FuzzedTensor(
                    name="x", size=("k0", "k1", "k2"), dim_parameter="d", roll_parameter="dim",
                    probability_contiguous=0.75, max_elements=128 * 1024,
                ),
                benchmark_utils.FuzzedTensor(
                    name="index", size=("m", "k1", "k2"),
                    tensor_constructor=lambda size, k0, **kwargs: torch.randint(0, k0, size=size),
                    dim_parameter="d", roll_parameter="dim",
                    probability_contiguous=0.75, max_elements=64 * 100000,
                ),
                benchmark_utils.FuzzedTensor(
                    name="out", size=("m", "k1", "k2"),
                    tensor_constructor=lambda size, **kwargs: torch.empty(size),
                    dim_parameter="d", roll_parameter="dim",
                    probability_contiguous=0.75, max_elements=64 * 100000,
                )
            ],
            seed=0
        )

    def take(self, n):
        """Label examples by index, and populate several global values."""
        for i, example in enumerate(super(GatherFuzzer, self).take(n)):
            example.globals["torch"] = torch
            example.globals["dim"] = example.metadata["dim"]
            yield benchmark_utils.Example(
                example.globals, f"[{i}]", example.metadata)



def subprocess_main(env: str, heuristic: str, result_file:str):
    # heuristic must be non-empty when passed to subprocess.
    heuristic = "" if heuristic == "N/A" else heuristic

    assert env in _ENVS
    assert hasattr(torch, "set_sg_heuristic") == (env == "branch")
    assert bool(heuristic) == (env == "branch")
    if heuristic:
        torch.set_sg_heuristic("gather", heuristic)

    timer = benchmark_utils.Timer(
        stmt="torch.gather(x, dim, index, out=out)",
        globals=GatherFuzzer(),
        label="gather",
        sub_label="gather" + (f" ({heuristic})" if heuristic else ""),
        env=env,
    )

    with open(result_file, "wb") as f:
        for i in timer.blocked_autorange(n=_NUM_RUNS):
            pickle.dump(i, f)


def _main():
    results = []
    for i, (env, heuristic) in enumerate(_CONFIGURATIONS * _REPEATS):
        results.extend(invoke_subprocess(env, heuristic))
        print(f"\r{i + 1} / {len(_CONFIGURATIONS) * _REPEATS}", end="")
        sys.stdout.flush()
    print()

    compare = benchmark_utils.Compare(results)
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()


def invoke_subprocess(env: str, heuristic: str):
    _, result_file = tempfile.mkstemp(suffix=".pkl")
    output = []
    proc = subprocess.run(
        f"source activate {env} && python -m examples.advanced "
        f"--context subprocess --env {env} --heuristic {heuristic} "
        f"--result_file {result_file}",
        stdout=subprocess.PIPE,
        shell=True,
    )

    try:
        if not proc.returncode:
            with open(result_file, "rb") as f:
                while True:
                    try:
                        output.append(pickle.load(f))
                    except EOFError:
                        break
    finally:
        os.remove(result_file)

    return output


def main():
    for env in _ENVS:
        result = subprocess.run(
            f"source activate {env}",
            stdout=subprocess.PIPE,
            shell=True,
        )
        if result.returncode != 0:
            raise ValueError(f"Failed to source environment `{env}`")

    _main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context", type=str,
        choices=("main", "subprocess"),
        default="main")
    parser.add_argument(
        "--env", type=str,
        choices=_ENVS,
        default=None)
    parser.add_argument(
        "--heuristic", type=str,
        choices=[i[1] for i in _CONFIGURATIONS],
        default="")
    parser.add_argument(
        "--result_file", type=str,
        default="")
    args = parser.parse_args()

    if args.context == "main":
        main()

    if args.context == "subprocess":
        subprocess_main(args.env, args.heuristic, args.result_file)
