import collections
import logging
from typing import List

#TODO(robieta): clean up relative imports
from . import common
from . import timer


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


def key_fn(m: common.Measurement):
    return m.label, m.sub_label, m.description, m.env, m.num_threads

def row_fn(m: common.Measurement):
    return r.num_threads, r.env, r.as_row_name


class Compare(object):
    def __init__(self, results: List[common.Measurement]):
        self._results = []
        self.extend_results(results)

    def extend_results(self, results):
        for r in results:
            if not isinstance(r, common.Measurement):
                raise ValueError("Expected an instance of `Measurement`, "
                                 f"got {type(r)} instead.")
        self._results.extend(results)

    def _render(self):
        results = self._merge_results()
        results = self._group_by_label(results)
        for group in results.values():
            self._layout(group)

    def _merge_results(self):
        grouped_results = collections.defaultdict(list)
        for r in self._results:
            grouped_results[key_fn(r)].append(r)

        output = []
        for (label, sub_label, description, env, num_threads), group in grouped_results.items():
            times = []
            for r in group:
                times.extend([t / r.number_per_run for t in r.times])
            unique_stmts = {r.stmt for r in group}
            if len(unique_stmts) != 1:
                logging.warning(
                    "Merged Examples with identical `label`, `sub_label`,\n"
                    "`description`, `env`, and `num_threads`, but different"
                    "`stmt`s:\n  " + "\n  ".join(unique_stmts))
            output.append(common.Measurement(
                number_per_run=1,
                times=times,
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description=description,
                env=env,
                stmt=unique_stmts.pop(),
                metadata=None,
            ))
        return output

    def _group_by_label(self, results):
        grouped_results = collections.defaultdict(list)
        for r in results:
            grouped_results[r.label].append(r)
        return grouped_results

    def _layout(self, results: List[common.Measurement]):
        assert len(set(r.label for r in results)) == 1
        label = results[0].label
        time_unit, time_scale = common.select_unit(min(r.median for r in results))

        rows = ordered_unique(results, row_fn)
        rows.sort(key=lambda args: args[:2])  # preserve stmt order
        columns = ordered_unique(results, lambda r: r.description)
        print(rows)
        print(columns)



    def print(self):
        self._render()
