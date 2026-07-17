"""
Tests for the recall-supervised data mix (the surviving piece of the
2026-07-13 signal round — the recall-rich mix is the default Track-A data):

  * tools/prepare_recall_mix.splice_facts_probe — the pure token splice behind
    the recall-supervised data mix (tested directly; the tool's module level
    imports nothing heavy).

(The round's other fixes — teacher distillation and read_init_scale — were
closed with the bolt-on line and removed from train.py, 2026-07-17.)

Run: /c/Users/henry/miniconda3/envs/cortex/python.exe -m pytest tests/test_signal_fixes.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.prepare_recall_mix import splice_facts_probe


class TestSpliceFactsProbe:
    ROW = 128

    def _row(self):
        return list(range(1000, 1000 + self.ROW))

    def test_row_length_preserved(self):
        out = splice_facts_probe(self._row(), [(10, [7, 7, 7])], [9, 9])
        assert len(out) == self.ROW

    def test_probe_at_tail(self):
        probe = [901, 902, 903]
        out = splice_facts_probe(self._row(), [(10, [7, 7])], probe)
        assert out[-3:] == probe

    def test_single_fact_at_exact_position(self):
        fact = [7, 7, 7, 7]
        out = splice_facts_probe(self._row(), [(20, fact)], [9])
        assert out[20:24] == fact
        assert out[:20] == self._row()[:20]           # prefix untouched
        assert out[24:30] == self._row()[20:26]       # suffix shifted right

    def test_multi_fact_drift_equals_lower_lengths(self):
        f1, f2 = [7, 7, 7], [8, 8]
        out = splice_facts_probe(self._row(), [(10, f1), (50, f2)], [9])
        assert out[10:13] == f1                        # lowest fact: exact pos
        # higher fact drifts right by len(f1)
        assert out[50 + len(f1): 50 + len(f1) + len(f2)] == f2

    def test_no_room_raises(self):
        with pytest.raises(AssertionError):
            splice_facts_probe(self._row(), [(self.ROW - 2, [7, 7, 7])],
                               [9] * 4)
