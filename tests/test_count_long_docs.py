"""
tools/count_long_docs.py -- the P0.4 half that sizes a long-document leg.

The script's entire value is that its arithmetic IS the packer's.  It predicts
how many rows prepare_pg19_dataset.py would emit from a corpus, and that
prediction decides whether a pack gets built at all -- so the thing worth
pinning is not "does the formula look right" but "does it agree with
stride_windows on every boundary case".  It already did not: at a remainder of
exactly row_len - 1 the document-end EOS fills the window and the packer emits
an all-ones mask, which the obvious floor-division reading counts as ragged.

A counter that drifts from the packer is worse than no counter, because the
number it produces is quoted in a corpus decision and nothing downstream
re-checks it.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from count_long_docs import pct, yield_rows              # noqa: E402
from prepare_pg19_dataset import stride_windows          # noqa: E402

EOS = 99


def _packer_counts(L, row_len):
    """What stride_windows actually emits: rows whose mask is all ones, and
    rows that carry padding."""
    wins = stride_windows(list(range(L)), row_len, EOS)
    full = sum(1 for _, m in wins if all(m))
    return full, len(wins) - full


class TestAgreesWithThePacker:

    def test_every_remainder_at_a_small_row_len(self):
        """Sweep a whole period plus change, so no remainder is untested."""
        RL = 17
        for L in range(0, 3 * RL + 2):
            f, r, _ = yield_rows([L], RL)
            assert (f, r) == _packer_counts(L, RL), f"L={L}"

    def test_the_remainder_that_the_eos_fills(self):
        """The case the obvious formula gets wrong: remainder == row_len - 1
        means the appended document-end EOS completes the window."""
        RL = 17
        f, r, _ = yield_rows([RL - 1], RL)
        assert (f, r) == (1, 0) == _packer_counts(RL - 1, RL)

    def test_exact_multiple_has_no_ragged_row(self):
        RL = 17
        for k in (1, 2, 5):
            f, r, _ = yield_rows([k * RL], RL)
            assert (f, r) == (k, 0) == _packer_counts(k * RL, RL)

    def test_at_the_real_row_len(self):
        """4,097 is what max_length 4096 actually packs at."""
        RL = 4097
        for L in (1, 1100, 4095, 4096, 4097, 4098, 8194, 100_000):
            f, r, _ = yield_rows([L], RL)
            assert (f, r) == _packer_counts(L, RL), f"L={L}"


class TestTheVolumeQuestion:

    def test_short_documents_yield_no_full_rows(self):
        """The finding the script exists to surface: a corpus of ~1.1k-token
        documents supplies ZERO rows to a doc-per-row pack at 4,096, however
        many tokens it holds."""
        full, ragged, kept = yield_rows([1100] * 1000, 4097)
        assert full == 0 and kept == 0
        assert ragged == 1000

    def test_books_yield_many_rows_each(self):
        full, _, kept = yield_rows([250_000] * 10, 4097)
        assert full == 10 * (250_000 // 4097)
        assert kept == full * 4097

    def test_totals_are_summed_not_averaged(self):
        """A 3-window document must contribute three rows -- counting documents
        over the threshold instead of windows is the hand-arithmetic mistake
        this script replaces."""
        mixed = [4097 * 3, 1100, 4097 * 2 + 5]
        full, _, _ = yield_rows(mixed, 4097)
        assert full == 3 + 0 + 2


class TestPercentiles:

    def test_orders_before_indexing(self):
        assert pct([9, 1, 5, 3, 7], 0.5) == 5

    def test_empty_is_zero_not_an_error(self):
        """The caller prints a table before it knows the corpus is non-empty."""
        assert pct([], 0.9) == 0

    def test_top_quantile_stays_in_range(self):
        assert pct([1, 2, 3], 1.0) == 3
