#!/bin/bash
# Gate A — the Tier-3 compat-branch grep.  Run on a PACE login node.
# CLEANUP_AUDIT.md §4 / cortex_next_phase_framework.md §10.  2026-09-15.
#
# WHAT IT DECIDES.  `prefix_pos="zero"` and `prefix_eos_reset=true` exist only
# to reproduce the cancelled 4vop2ym8 run, which was rm -rf'd.  Both keys are in
# the 16-flag persist list, so they must keep LOADING; the branches need not
# keep working.  Empty result -> remove the branches from prefix_pack /
# prefix_unpack / _carried_state and keep the keys.  Any hit -> leave both alone
# and record which checkpoint depends on them.
#
# WHY THIS IS A SCRIPT AND NOT THE ONE-LINER IN THE AUDIT.  The audit's glob is
#
#     $SCRATCH/cortex-*/*/config.json
#
# and it is ONE LEVEL TOO SHALLOW, so it matches nothing for a structural reason
# rather than an evidential one.  train.py writes config.json only through
# save_model_only -> save_pretrained(f"{out_path}/{run_name}/{chkpt_name}")
# (train.py:531-534), i.e. at
#
#     $SCRATCH/<out_path>/<run_name>/<chkpt_name>/config.json      # THREE levels
#
# save_checkpoint (train.py:536) writes only chkpt.pt into checkpoint_<step>/ —
# no config.json at all.  So the audit's glob expands to
# $SCRATCH/cortex-control/c-chunked/config.json and friends, none of which
# exist, and grep returns empty on EVERY cluster regardless of what is stored.
# Acting on that empty result would delete a live compat path on no evidence —
# bug class 2 (find where a value is CONSUMED, not where it is set), which has
# now voided or distorted four runs.
#
# Hence: this script counts what it scanned FIRST, NAMES EVERY ROOT IT COULD NOT
# OPEN, and refuses to report a verdict if the scan set is empty.  A root that
# silently does not exist is the same false negative one level up.
#
# SCOPE.  config.json is the branches' load path — cortex_graft.py:266-270 reads
# both flags off the HF config via getattr — so config.json coverage is the
# right coverage.  That includes the BASE checkpoints under $SCRATCH/ckpts,
# because --model_name is what a resume link builds its model from; chkpt.pt
# restores weights into a model that was already constructed from that config.
#
# Usage:  bash pace/check_tier3_compat.sh
set -u

: "${SCRATCH:?SCRATCH is not set — run this on a PACE login node}"

PATTERN='"prefix_pos": *"zero"\|"prefix_eos_reset": *true'

echo "=== Gate A — Tier-3 compat-branch grep ==="
echo "scratch: $SCRATCH"
echo

# Every config.json under a run tree or a base-checkpoint tree.  Roots are
# resolved one at a time so a missing one is REPORTED rather than swallowed by
# find's stderr — see the note above.
ROOTS=()
MISSING=()
for r in "$SCRATCH"/cortex-* "$SCRATCH"/ckpts; do
    if [ -d "$r" ]; then ROOTS+=("$r"); else MISSING+=("$r"); fi
done

if [ "${#ROOTS[@]}" -eq 0 ]; then
    echo "REFUSING TO REPORT A VERDICT: none of the expected roots exist."
    printf '  missing: %s\n' "${MISSING[@]}" | sed "s|$SCRATCH|\$SCRATCH|"
    exit 2
fi

mapfile -t FILES < <(
    find "${ROOTS[@]}" -maxdepth 4 -name config.json -type f 2>/dev/null | sort
)

echo "roots scanned: ${#ROOTS[@]}"
printf '  %s\n' "${ROOTS[@]}" | sed "s|$SCRATCH|\$SCRATCH|"
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo
    echo "ROOTS NOT PRESENT (not scanned — confirm this is expected):"
    printf '  %s\n' "${MISSING[@]}" | sed "s|$SCRATCH|\$SCRATCH|"
    echo "  If base checkpoints live elsewhere (--model_name is a path RELATIVE"
    echo "  to the submit dir, so 'ckpts/...' may resolve inside the repo or"
    echo "  through a symlink), re-run with that directory added, e.g."
    echo "    find -L \"\$(readlink -f ckpts)\" -maxdepth 2 -name config.json \\"
    echo "      -exec grep -l '$PATTERN' {} +"
fi
echo
echo "config.json files scanned: ${#FILES[@]}"
if [ "${#FILES[@]}" -eq 0 ]; then
    echo
    echo "REFUSING TO REPORT A VERDICT: the scan set is empty."
    echo "That is the failure mode this script exists to prevent — an empty grep"
    echo "that means 'nothing was looked at', not 'nothing depends on it'."
    echo "Check that \$SCRATCH/cortex-* and \$SCRATCH/ckpts exist and hold runs."
    exit 2
fi

printf '  %s\n' "${FILES[@]}" | sed "s|$SCRATCH|\$SCRATCH|"
echo

HITS=$(grep -l "$PATTERN" "${FILES[@]}" 2>/dev/null)

if [ -z "$HITS" ]; then
    echo "VERDICT: CLEAR — no surviving checkpoint carries prefix_pos=zero or"
    echo "prefix_eos_reset=true across ${#FILES[@]} config.json files."
    echo
    echo "Action: remove the compat branches, KEEP both keys."
    echo "  cortex_graft.py:403  prefix_pack        self.prefix_pos == 'tail' conjunct"
    echo "  cortex_graft.py:435  prefix_unpack      _valid_write zeroing under prefix_eos_reset"
    echo "  cortex_graft.py:491  _carried_state     _write_reset zeroing under prefix_eos_reset"
    echo "Keep the loads at cortex_graft.py:266-270 and the persist list at train.py:875,"
    echo "and assert the modern value instead of branching on it."
    echo "Tests that pin the OLD behaviour and must be rewritten as assertions:"
    echo "  tests/test_prefix_pack_flags.py  TestPrefixPos / TestPrefixEosReset"
    echo "  tests/test_prefix_memory.py:326-361"
    echo "  tests/test_smoke_prefix_real.py:77"
    exit 0
else
    echo "VERDICT: DEPENDENT CHECKPOINTS EXIST — leave both branches alone."
    echo "$HITS" | sed "s|$SCRATCH|\$SCRATCH|"
    echo
    echo "Record these in CLEANUP_AUDIT.md §4 as the reason the branches stay."
    exit 1
fi
