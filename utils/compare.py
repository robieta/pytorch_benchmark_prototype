import collections
import itertools as it
import logging
from typing import List, Tuple

import numpy as np

# TODO(robieta): clean up relative imports
from . import common, timer


_SIGNIFICANT_FIGURES = 3


def ordered_unique(elements, key_fn=None):
    if key_fn is None:
        key_fn = lambda x: x
    output = []
    seen = set()
    for e in elements:
        key = key_fn(e)
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output


def group_fn(m: common.Measurement):
    return m.label, m.sub_label, m.description, m.env, m.num_threads


# Classes to separate internal bookkeeping from what is rendered.
class Column(object):
    def __init__(
        self,
        grouped_results: List[List[common.Measurement]],
        time_scale: float
    ):
        self._grouped_results = grouped_results
        self._flat_results = list(it.chain(*grouped_results))
        self._time_scale = time_scale
        leading_digits = [
            int(np.log10(r.median / self._time_scale) // 1)
            for r in self._flat_results
        ]
        unit_digits = max(leading_digits)
        decimal_digits = max(0, _SIGNIFICANT_FIGURES - min(leading_digits))
        length = unit_digits + decimal_digits + (1 if decimal_digits else 0)
        self._template = f"{{:>{length}.{decimal_digits}f}}"


class Row(object):
    def register_columns(self, columns: Tuple[Column]):
        pass


class DataRow(Row):
    def __init__(self, results, row_group, render_env, time_scale):
        super(DataRow, self).__init__()
        self._results = results
        self._row_group = row_group
        self._render_env = render_env
        self._time_scale = time_scale
        self._columns = None

    def register_columns(self, columns: Tuple[Column]):
        self._columns = columns


class ThreadCountRow(Row):
    def __init__(self, num_threads):
        super(ThreadCountRow, self).__init__()
        self._num_threads = num_threads


class Table(object):
    def __init__(self, results: List[common.Measurement]):
        assert len(set(r.label for r in results)) == 1
        assert len({group_fn(r) for r in results}) == len(results)

        self.results = results
        self.label = results[0].label
        self.time_unit, self.time_scale = common.select_unit(
            min(r.median for r in results)
        )

        self.row_keys = ordered_unique(results, self.row_fn)
        self.row_keys.sort(key=lambda args: args[:2])  # preserve stmt order
        self.column_keys = ordered_unique(results, self.col_fn)
        self.rows, self.columns = self.populate_rows_and_columns()

    @staticmethod
    def row_fn(m: common.Measurement):
        return m.num_threads, m.env, m.as_row_name

    @staticmethod
    def col_fn(m: common.Measurement):
        return m.description

    def populate_rows_and_columns(self):
        rows, columns = [], []

        ordered_results = [[None for _ in self.column_keys] for _ in self.row_keys]
        row_position = {key: i for i, key in enumerate(self.row_keys)}
        col_position = {key: i for i, key in enumerate(self.column_keys)}
        for r in self.results:
            i = row_position[self.row_fn(r)]
            j = col_position[self.col_fn(r)]
            ordered_results[i][j] = r

        prior_num_threads = -1
        prior_env = ""
        row_group = 0
        rows_by_group = []
        for (num_threads, env, _), row in zip(self.row_keys, ordered_results):
            if num_threads != prior_num_threads:
                prior_num_threads = num_threads
                prior_env = ""
                row_group += 1
                rows_by_group.append([])
                rows.append(ThreadCountRow(num_threads=num_threads))
            rows.append(
                DataRow(
                    results=row,
                    row_group=row_group,
                    render_env=(env != prior_env),
                    time_scale=self.time_scale,
                )
            )
            rows_by_group[-1].append(row)
            prior_env = env

        for i in range(len(self.column_keys)):
            grouped_results = [tuple(row[i] for row in g) for g in rows_by_group]
            column = Column(grouped_results=grouped_results, time_scale=self.time_scale)
            columns.append(column)

        rows, columns = tuple(rows), tuple(columns)
        for r in rows:
            r.register_columns(columns)
        return rows, columns


class Compare(object):
    def __init__(self, results: List[common.Measurement]):
        self._results = []
        self.extend_results(results)

    def extend_results(self, results):
        for r in results:
            if not isinstance(r, common.Measurement):
                raise ValueError(
                    "Expected an instance of `Measurement`, " f"got {type(r)} instead."
                )
        self._results.extend(results)

    def _render(self):
        results = self._merge_results()
        results = self._group_by_label(results)
        for group in results.values():
            self._layout(group)

    def _merge_results(self):
        grouped_results = collections.defaultdict(list)
        for r in self._results:
            grouped_results[group_fn(r)].append(r)

        output = []
        for key, group in grouped_results.items():
            label, sub_label, description, env, num_threads = key
            times = []
            for r in group:
                times.extend([t / r.number_per_run for t in r.times])
            unique_stmts = {r.stmt for r in group}
            if len(unique_stmts) != 1:
                logging.warning(
                    "Merged Examples with identical `label`, `sub_label`,\n"
                    "`description`, `env`, and `num_threads`, but different"
                    "`stmt`s:\n  " + "\n  ".join(unique_stmts)
                )
            output.append(
                common.Measurement(
                    number_per_run=1,
                    times=times,
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description=description,
                    env=env,
                    stmt=unique_stmts.pop(),
                    metadata=None,
                )
            )
        return output

    def _group_by_label(self, results):
        grouped_results = collections.defaultdict(list)
        for r in results:
            grouped_results[r.label].append(r)
        return grouped_results

    def _layout(self, results: List[common.Measurement]):
        table = Table(results)

    def print(self):
        self._render()
