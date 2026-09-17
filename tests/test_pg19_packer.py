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

import io
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


class TestTheValidationPack:
    """The 50-row cap, and the launcher change that lifts it.

    PG-19's validation split is 50 books.  Under the old truncating packer that
    was 50 rows, so every eval reading `data/pg19_olmo_val_len4096` silently ran
    at n=50 however large --max_examples was -- the P2.1 sweep asked for 100 and
    got 50 on all 18 cells.  The striding fix already exists; what was missing
    was any way to point it at a split other than train, and any warning when
    the pack binds instead of the flag.
    """

    def _sbatch(self, name):
        path = os.path.join(REPO, "pace", name)
        return io.open(path, encoding="utf-8").read()

    def test_fifty_books_stride_into_hundreds_of_rows(self):
        """The arithmetic the fix rests on, at PG-19's real shape: books average
        ~69k tokens, so each one is ~17 windows at row_len 4096 rather than 1."""
        rows = 0
        for _ in range(50):
            rows += len(stride_windows(list(range(69_000)), row_len=4096, eos=EOS))
        assert rows > 800, "striding 50 books must beat the 50-row pack by >10x"

    def test_the_launcher_can_ask_for_a_split(self):
        s = self._sbatch("prepare_pg19_pack.sbatch")
        assert "SPLIT=${SPLIT:-train}" in s
        assert '--split "$SPLIT"' in s

    def test_a_non_train_split_cannot_land_on_the_train_packs_name(self):
        """Two artifacts under one name is the ARM_DATA failure; the existing
        header already refuses to overwrite, and the derived default keeps a
        validation pack from ever needing that refusal."""
        s = self._sbatch("prepare_pg19_pack.sbatch")
        assert 'data/pg19_olmo_${SPLIT}_len${MAX_LENGTH}_strided' in s
        assert 'data/pg19_olmo_len${MAX_LENGTH}_strided' in s

    def test_the_deciding_cells_print_the_row_count(self):
        """n=50 was discovered after 18 cells had run.  The cell now says what
        the pack holds, and says so LOUDER when the flag over-promises."""
        s = self._sbatch("eval_deciding.sbatch")
        assert "PACK_ROWS" in s
        assert "is a CEILING" in s
        assert "the pack binds, not the flag" in s
