"""tools/prepare_pg19_dataset.py's striding, after the 2026-09-14 truncation fix.

The bug this pins was invisible: the old packer emitted `ids[:row_len]`, one row
per document, under a `char_cap = row_len * 8` that stopped tokenizing at ~32k
characters.  The result was a perfectly valid HF dataset that happened to hold
roughly the FIRST 1% OF EACH BOOK — 28,602 rows / ~117M tokens against PG-19's
~2.5B — and every "PG-19 cannot supply the volume" decision in the project was
downstream of that row count.

Nothing about a truncating packer looks wrong from the outside: the rows are the
right shape, the masks are valid, training runs.  So the properties worth
asserting are the ones that distinguish "strided" from "truncated" at all.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from prepare_pg19_dataset import stride_windows  # noqa: E402

EOS = 999


class TestStriding:

    def test_a_long_document_becomes_many_rows(self):
        # The whole fix.  Under the old packer this was 1.
        rows = stride_windows(list(range(1000)), row_len=100, eos=EOS)
        assert len(rows) == 10

    def test_no_token_is_dropped(self):
        ids = list(range(1000))
        rows = stride_windows(ids, row_len=100, eos=EOS)
        seen = [t for win, mask in rows for t, m in zip(win, mask) if m == 1]
        assert seen == ids

    def test_no_token_is_repeated(self):
        # Non-overlapping windows: an epoch must not train twice on the same
        # span, which a stride shorter than row_len would silently cause.
        ids = list(range(1000))
        rows = stride_windows(ids, row_len=100, eos=EOS)
        seen = [t for win, mask in rows for t, m in zip(win, mask) if m == 1]
        assert len(seen) == len(set(seen))

    def test_windows_are_contiguous_and_in_order(self):
        # Each row must be ONE continuous span of the document — the property
        # the cross-chunk carry depends on.  Shuffled or strided-with-gaps rows
        # would still pass a row-count check.
        rows = stride_windows(list(range(1000)), row_len=100, eos=EOS)
        for i, (win, _) in enumerate(rows):
            assert win == list(range(i * 100, (i + 1) * 100))

    def test_every_row_is_exactly_row_len(self):
        rows = stride_windows(list(range(1050)), row_len=100, eos=EOS)
        assert all(len(w) == 100 and len(m) == 100 for w, m in rows)


class TestRaggedTail:

    def test_tail_is_eos_marked_and_padded(self):
        rows = stride_windows(list(range(150)), row_len=100, eos=EOS)
        assert len(rows) == 2
        win, mask = rows[1]
        assert win[:50] == list(range(100, 150))
        assert win[50] == EOS, "no document-end marker for eos_from_tokens"
        assert mask[:51] == [1] * 51
        assert mask[51:] == [0] * 49
        assert set(win[51:]) == {EOS}

    def test_pads_are_masked_out_of_the_loss(self):
        _, mask = stride_windows(list(range(150)), row_len=100, eos=EOS)[1]
        # 50 real tokens + 1 EOS marker; the rest is padding.
        assert sum(mask) == 51

    def test_a_document_shorter_than_the_window_is_one_padded_row(self):
        rows = stride_windows(list(range(30)), row_len=100, eos=EOS)
        assert len(rows) == 1
        win, mask = rows[0]
        assert sum(mask) == 31 and win[30] == EOS

    def test_an_exact_multiple_leaves_no_ragged_row(self):
        rows = stride_windows(list(range(300)), row_len=100, eos=EOS)
        assert len(rows) == 3
        assert all(sum(m) == 100 for _, m in rows)

    def test_an_empty_document_yields_nothing(self):
        assert stride_windows([], row_len=100, eos=EOS) == []


class TestRaggedShare:
    """Why chunk_loss_reduction had to default to "token" BEFORE this pack was
    built: striding makes the last window of every book ragged."""

    def test_one_ragged_row_per_document(self):
        # 1 in 8 rows at ~8 rows/book, which is the ~12% the train.py comment
        # predicts — against ~1% under the truncating packer.
        rows = stride_windows(list(range(750)), row_len=100, eos=EOS)
        ragged = [m for _, m in rows if sum(m) != 100]
        assert len(ragged) == 1
        assert len(rows) == 8
