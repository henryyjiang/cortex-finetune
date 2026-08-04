"""
Analytic FLOP accounting for compute-matched benchmarking (2026-08-03).

WHY ANALYTIC.  The obvious way to compute-match cortex against a baseline is to
measure what the eval harness actually costs.  That is wrong here, and wrong in
a direction that would manufacture a result: until the KV cache landed, the
generation loop re-forwarded the entire prefix for every token, so decoding cost
grew as O(G * (S+G)) instead of O(G * S + G^2/2).  Compute-matching on measured
cost would have handed the CoT-heavy baseline a penalty of ~2 orders of
magnitude that exists in the harness and not in the architecture.  Wall-clock is
worse still (kernel efficiency, batch size, GPU model).  So the matching is done
from a closed-form cost model over the architecture, and the harness is free to
be as fast or slow as it likes.

WHAT IS BEING MATCHED.  The claim under test is a compute-ALLOCATION claim: at a
fixed budget per query, is it better to spend on latent depth plus cross-chunk
memory, or on explicit chain-of-thought tokens in context?  So the unit is
FLOPs per query, and the free variable on the baseline side is how many CoT
tokens it is allowed to emit.  `match_cot_budget` solves for exactly that.

WHAT IS COUNTED.  Dense matmuls (attention projections, gated MLP, the recurrent
adapter, the head) plus the quadratic attention-score term.  Norms, RoPE,
softmax and residuals are omitted — they are O(d) per token against O(d^2), i.e.
well under a percent, and they are omitted IDENTICALLY on both sides, so they
cannot bias a ratio.  Backward is not modelled: this is inference only.

The cortex-specific costs that a naive model would miss, and that are the whole
point of doing this properly:
  * the packed sequence is longer than the token sequence — n_pre carried
    vectors are prepended and (when writing) n_sum summary slots appended, and
    every one of them pays full prelude + T*core + coda;
  * the recurrent block is paid T times per token, and the adapter with it;
  * chunking changes the attention term from one S^2 over the whole context to
    n_chunks smaller ones, which is the entire long-context cost argument and
    is invisible if you only count parameters.

Usage:
    # cost of one GSM8K-style query under a cortex config
    python evals/flops.py --model_name ckpts/olmo-retrofit-cortex \
        --prompt_tokens 900 --gen_tokens 256 --T 8

    # how many CoT tokens the baseline gets for the same budget
    python evals/flops.py --model_name ckpts/olmo-retrofit-cortex \
        --prompt_tokens 900 --gen_tokens 256 --T 8 --match_baseline_layers 16
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Architecture spec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Everything the cost model needs.  `n_core`/`n_prelude`/`n_coda` describe
    the recurrent arrangement; a plain transformer baseline is expressed as
    n_prelude = n_layers, n_core = 0, n_coda = 0 (so T is irrelevant and the
    adapter is never paid)."""
    d: int                  # n_embd
    n_head: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    n_prelude: int = 0
    n_core: int = 0
    n_coda: int = 0
    name: str = "model"

    @classmethod
    def from_config(cls, cfg, name: str = "cortex") -> "ModelSpec":
        d = int(cfg.n_embd)
        n_head = int(cfg.num_attention_heads)
        return cls(
            d=d,
            n_head=n_head,
            n_kv_heads=int(cfg.num_key_value_heads),
            head_dim=int(getattr(cfg, "head_dim", d // n_head)),
            intermediate=int(cfg.intermediate_size),
            vocab=int(getattr(cfg, "padded_vocab_size", getattr(cfg, "vocab_size", 0))),
            n_prelude=int(cfg.n_layers_in_prelude),
            n_core=int(cfg.n_layers_in_recurrent_block),
            n_coda=int(cfg.n_layers_in_coda),
            name=name,
        )

    @classmethod
    def dense_baseline(cls, spec: "ModelSpec", n_layers: int,
                       name: str = "baseline") -> "ModelSpec":
        """A non-recurrent transformer of the same width as `spec`.  Used for
        the control arm: same d/heads/MLP, n_layers stacked once, no adapter."""
        return cls(d=spec.d, n_head=spec.n_head, n_kv_heads=spec.n_kv_heads,
                   head_dim=spec.head_dim, intermediate=spec.intermediate,
                   vocab=spec.vocab, n_prelude=n_layers, n_core=0, n_coda=0,
                   name=name)

    # -- per-token, per-layer costs (matmuls only) --------------------------

    @property
    def attn_proj_flops(self) -> int:
        """Wqkv + out proj.  2*params, with GQA's smaller K/V correctly sized."""
        q = self.n_head * self.head_dim
        kv = self.n_kv_heads * self.head_dim
        return 2 * self.d * (q + 2 * kv) + 2 * q * self.d

    @property
    def mlp_flops(self) -> int:
        """GatedMLP: fc is Linear(d, 2*I) then proj is Linear(I, d)."""
        return 2 * self.d * (2 * self.intermediate) + 2 * self.intermediate * self.d

    @property
    def layer_flops(self) -> int:
        return self.attn_proj_flops + self.mlp_flops

    @property
    def adapter_flops(self) -> int:
        """Linear(2d, d), paid once per recurrent step per token."""
        return 4 * self.d * self.d if self.n_core else 0

    @property
    def head_flops(self) -> int:
        return 2 * self.d * self.vocab

    def score_flops(self, n_keys: int) -> int:
        """QK^T + AV for ONE query attending to n_keys keys, one layer."""
        return 4 * self.n_head * self.head_dim * n_keys

    def depth(self, T: int) -> int:
        return self.n_prelude + self.n_core * T + self.n_coda


# ---------------------------------------------------------------------------
# Cost of one forward
# ---------------------------------------------------------------------------

def forward_flops(spec: ModelSpec, n_tokens: int, T: int = 1,
                  n_pre: int = 0, n_sum: int = 0,
                  past_keys: int = 0, head_tokens: Optional[int] = None) -> int:
    """FLOPs for one forward over `n_tokens` new positions.

    n_pre / n_sum   packed carry and summary columns (cortex prefix memory).
                    They are real positions: they pay every layer.
    past_keys       keys already in the KV cache that these queries attend to
                    (0 for an uncached full forward).
    head_tokens     positions the lm_head is applied to (default: all real
                    tokens; pass 1 for a cached decode step).
    """
    packed = n_pre + n_tokens + n_sum
    n_layers = spec.depth(T)

    # Dense per-position work, paid at every layer of the unrolled depth.
    total = packed * n_layers * spec.layer_flops
    total += packed * T * spec.adapter_flops

    # Attention scores.  Query i (0-indexed within this forward) attends to
    # past_keys + i + 1 keys under the causal mask, so the sum over the packed
    # block is packed*past_keys + packed*(packed+1)/2.
    keys = packed * past_keys + packed * (packed + 1) // 2
    total += n_layers * spec.score_flops(keys)   # linear in key-visits

    ht = n_tokens if head_tokens is None else head_tokens
    total += ht * spec.head_flops
    return int(total)


def query_flops(spec: ModelSpec, prompt_tokens: int, gen_tokens: int,
                T: int = 1, n_chunks: int = 1, carry_vecs: int = 0,
                summary_vecs: int = 0, passes: int = 1,
                cached: bool = True) -> dict:
    """FLOPs for one benchmark query, split into ingest and generate.

    prompt_tokens   context read before generating (few-shot prompt, retrieved
                    documents, the long-context haystack).
    n_chunks        how the prompt is segmented.  Chunking is what makes the
                    attention term linear-ish in context instead of quadratic:
                    n_chunks * (S/n_chunks)^2 = S^2/n_chunks.
    carry_vecs      vectors prepended per chunk.  For PrefixAccumBuffer this
                    GROWS chunk to chunk (0, n_vec, 2*n_vec, ... capped at
                    accum_max), which is modelled here rather than assumed flat.
    summary_vecs    slots appended per ingest chunk (0 during generation).
    passes          full forward passes over the prompt before decoding
                    (ccot_prime / --ccot_passes).  Each is a full ingest.
    cached          KV-cached decoding (each generated token attends to the
                    growing prefix once).  False reproduces the old
                    re-forward-everything loop, for showing the gap.
    """
    per_chunk = max(prompt_tokens // max(n_chunks, 1), 1)
    cap = carry_vecs if carry_vecs else summary_vecs
    ingest = 0
    carried = 0
    for _ in range(max(passes, 1)):
        carried = 0
        for _ in range(max(n_chunks, 1)):
            ingest += forward_flops(spec, per_chunk, T, n_pre=carried,
                                    n_sum=summary_vecs, head_tokens=0)
            if summary_vecs:
                # PrefixAccumBuffer appends n_vec rows per chunk, FIFO-trimmed
                # at accum_max — so the carry is NOT flat across chunks.
                carried = min(carried + summary_vecs, cap)

    # Decoding: whatever the ingest left resident is in the cache throughout.
    resident = carried
    generate = 0
    if cached:
        # Prefill is counted in `ingest` only when the model re-reads the
        # prompt as chunks; the last chunk's KV is what decoding attends to.
        for g in range(gen_tokens):
            generate += forward_flops(spec, 1, T, n_pre=0, n_sum=0,
                                      past_keys=resident + per_chunk + g,
                                      head_tokens=1)
    else:
        for g in range(gen_tokens):
            generate += forward_flops(spec, per_chunk + g, T, n_pre=resident,
                                      n_sum=0, head_tokens=1)

    return {"ingest": int(ingest), "generate": int(generate),
            "total": int(ingest + generate), "depth": spec.depth(T)}


# ---------------------------------------------------------------------------
# The matching itself
# ---------------------------------------------------------------------------

def match_cot_budget(baseline: ModelSpec, budget: int, prompt_tokens: int,
                     max_tokens: int = 32768) -> int:
    """Largest number of generated tokens the baseline can afford inside
    `budget` FLOPs.  This is the compute-matched control: cortex gets depth and
    memory, the baseline gets to think longer in tokens, and both cost the same.

    Monotone in gen_tokens, so a bisection is exact."""
    def cost(g: int) -> int:
        return query_flops(baseline, prompt_tokens, g, T=1, n_chunks=1,
                           cached=True)["total"]

    if cost(0) > budget:
        return 0
    lo, hi = 0, 1
    while hi < max_tokens and cost(hi) <= budget:
        lo, hi = hi, hi * 2
    hi = min(hi, max_tokens)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Analytic FLOP accounting / compute matching")
    p.add_argument("--model_name", default=None,
                   help="Graft-prepared dir to read the architecture from. "
                        "Omit to use --d/--layers etc. directly.")
    p.add_argument("--prompt_tokens", type=int, default=900)
    p.add_argument("--gen_tokens",    type=int, default=256)
    p.add_argument("--T",             type=int, default=8)
    p.add_argument("--n_chunks",      type=int, default=1)
    p.add_argument("--carry_vecs",    type=int, default=128,
                   help="accum_max: the cap on resident carried vectors")
    p.add_argument("--summary_vecs",  type=int, default=32,
                   help="accum_vecs: slots appended per ingest chunk")
    p.add_argument("--passes",        type=int, default=1,
                   help="ccot_passes + 1 (full prompt passes before decoding)")
    p.add_argument("--match_baseline_layers", type=int, default=None,
                   help="Solve for the CoT budget of a dense N-layer baseline "
                        "of the same width at equal total FLOPs.")
    p.add_argument("--show_uncached", action="store_true",
                   help="Also report the pre-KV-cache harness cost, to show "
                        "why measured-cost matching would have been wrong.")
    p.add_argument("--out", default=None, help="Write the report as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_name:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        spec = ModelSpec.from_config(cfg)
    else:
        raise SystemExit("--model_name is required (the architecture is read "
                         "from the checkpoint config)")

    cortex = query_flops(spec, args.prompt_tokens, args.gen_tokens, T=args.T,
                         n_chunks=args.n_chunks, carry_vecs=args.carry_vecs,
                         summary_vecs=args.summary_vecs, passes=args.passes)

    report = {"spec": asdict(spec), "cortex": cortex,
              "config": {"prompt_tokens": args.prompt_tokens,
                         "gen_tokens": args.gen_tokens, "T": args.T,
                         "n_chunks": args.n_chunks, "passes": args.passes}}

    print(f"\n{spec.name}: d={spec.d} heads={spec.n_head}/{spec.n_kv_heads} "
          f"I={spec.intermediate} depth={cortex['depth']} (T={args.T})")
    print(f"  ingest   {cortex['ingest']:>18,}")
    print(f"  generate {cortex['generate']:>18,}")
    print(f"  TOTAL    {cortex['total']:>18,}   ({cortex['total']/1e12:.2f} TFLOPs)")

    if args.show_uncached:
        un = query_flops(spec, args.prompt_tokens, args.gen_tokens, T=args.T,
                         n_chunks=args.n_chunks, carry_vecs=args.carry_vecs,
                         summary_vecs=args.summary_vecs, passes=args.passes,
                         cached=False)
        report["cortex_uncached_harness"] = un
        print(f"\n  uncached harness TOTAL {un['total']:>14,}  "
              f"({un['total'] / max(cortex['total'], 1):.1f}x the cached cost "
              f"— this factor is harness, not architecture)")

    if args.match_baseline_layers:
        base = ModelSpec.dense_baseline(spec, args.match_baseline_layers)
        g = match_cot_budget(base, cortex["total"], args.prompt_tokens)
        base_cost = query_flops(base, args.prompt_tokens, g, T=1, n_chunks=1)
        report["baseline"] = {"spec": asdict(base), "cot_tokens": g,
                              "cost": base_cost}
        print(f"\ncompute-matched baseline ({args.match_baseline_layers} dense layers):")
        print(f"  CoT token budget {g:>14,}   (cortex generates {args.gen_tokens:,})")
        print(f"  cost             {base_cost['total']:>14,}   "
              f"({base_cost['total'] / max(cortex['total'], 1):.4f}x cortex)")
        print(f"\n  -> run the baseline with --max_new_tokens {g}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
