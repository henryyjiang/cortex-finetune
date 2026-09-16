"""
Shared model-loading + interface helpers for the cortex evals on the raven
(RavenForCausalLM) model.

These adapt the cortex-main eval scripts (written against CortexGPT) to the
retrofitting-recurrence model with three thin shims:

  load_checkpoint(checkpoint, model_name, memory_slots, dtype, device)
      Load a raven model via from_pretrained(model_name, trust_remote_code).
      `model_name` should be a graft-prepared model dir (see
      tools/prepare_cortex_checkpoint.py) so the grafted modeling file + memory
      flags are active; passing memory_slots forces use_memory on the config.
      `checkpoint` (optional) is a torch .pt saved by train.py whose ["model"]
      state_dict is overlaid with strict=False (finetuned weights).
      Returns (model, config); config.mean_recurrence is the default eval T.

  has_cross_state(model) -> bool
      True if the grafted model carries cross-segment memory (M_cross / DirectCCoT).

  to_num_steps(T) -> Optional[torch.Tensor]
      Eval recurrence depth → raven num_steps. T iterations, all no-grad
      (eval runs under torch.no_grad anyway).  None → model uses its config
      mean_recurrence.

NOTE: run evals from the repo root so the grafted modeling file's
`from cortex_graft import ...` resolves.  Use the cortex-retro env
(transformers ~4.51); the cortex env's transformers 5.x cannot load the base.
"""
from __future__ import annotations

import os
import sys
import zlib
from typing import Optional

import torch

# Allow importing cortex_graft / cortex_memory from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _unwrap(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def has_cross_state(model) -> bool:
    cortex = getattr(_unwrap(model), "cortex", None)
    return cortex is not None and cortex.has_cross_state


def accumulating_buffer(model):
    """The model's WRITE-ONCE accumulating carry buffer, or None.

    The slice ablation and the buffer diagnostic both need per-chunk rows to
    stay separable: each chunk appends exactly `n_vec` rows and nothing ever
    rewrites an older one.  That holds for the prefix ACCUM buffer (and for the
    retired AccumCCoT), and NOT for either gated buffer, whose merge mixes the
    whole state — there the k-th chunk's contribution cannot be recovered, so
    both tools must refuse rather than report a meaningless number.

    Duck-typed on the append-buffer contract (n_vec + max_vecs) so it keeps
    working for old checkpoints whose buffer class has since been retired.
    """
    cortex = getattr(_unwrap(model), "cortex", None)
    if cortex is None:
        return None
    for attr in ("prefix", "accum"):
        buf = getattr(cortex, attr, None)
        if buf is not None and hasattr(buf, "max_vecs") and hasattr(buf, "n_vec"):
            return buf
    return None


def to_num_steps(T: Optional[int]):
    if T is None:
        return None
    return torch.tensor([int(T), 0])


def parse_config_overrides(items) -> dict:
    """["KEY=VALUE", ...] -> a TYPED dict for `load_checkpoint(config_overrides=)`.

    Booleans and ints are parsed, and that is the whole reason this is a
    function rather than a dict comprehension at each call site: a config flag
    that arrives as the STRING "false" is TRUTHY, so `latent_carry=false` would
    turn an intended E-only run into a dual-channel one with no symptom at all.

    Shared by tools/prelaunch_final.py and evals/diag_dual_channel_walk.py so
    the two cannot parse the same flag differently.
    """
    out = {}
    for it in items or ():
        if "=" not in it:
            raise SystemExit(f"expected KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        low = v.strip().lower()
        if low in ("true", "false"):
            out[k] = (low == "true")
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def explain_missing_cortex(cfg, overrides) -> str:
    """Why is `model.cortex` None?  Name the cause instead of the symptom.

    `use_memory` is the MASTER SWITCH -- cortex_graft.memory_enabled reads that
    and nothing else -- and a graft-prepared BASE dir may carry no cortex flags
    at all.  So overriding prefix_memory/accum_vecs/gate_slots without it builds
    nothing, the checkpoint's cortex tensors load as "unexpected keys" and are
    dropped, and the run silently becomes a no-memory baseline.  That is how
    job 13266470 failed three post-load steps at once, each reporting only
    "no prefix buffer".
    """
    on = bool(getattr(cfg, "use_memory", False))
    geom = [k for k in ("prefix_memory", "accum_vecs", "accum_max", "gate_slots",
                        "latent_carry")
            if (overrides or {}).get(k) is not None or hasattr(cfg, k)]
    if not on and geom:
        return ("config.use_memory is FALSE/absent while memory geometry was "
                f"given ({', '.join(sorted(geom))}).  use_memory is the master "
                "switch the graft reads; without it nothing is built and the "
                "checkpoint's cortex tensors are dropped as unexpected keys.  "
                "Add: --set use_memory=true --set memory_slots=0")
    if not on:
        return ("config.use_memory is FALSE/absent, so this model has no cortex "
                "at all.  Add --set use_memory=true, or point --model_name at a "
                "graft-prepared dir whose config.json carries the flags.")
    return ("use_memory is set but no prefix buffer was built -- check "
            "--set prefix_memory=accum|gated, and that cortex_graft imported "
            "(run from the repo root).")


def load_checkpoint(
    checkpoint: Optional[str],
    model_name: str,
    memory_slots: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
    accum_max: Optional[int] = None,
    config_overrides: Optional[dict] = None,
):
    # A --checkpoint that is not a file used to be IGNORED here, and the run
    # went ahead on the base weights with nothing in the log saying so -- the
    # exact silent class this file's other guards exist for.  A checkpoint DIR
    # is the natural thing to pass (train.py writes checkpoint_<step>/chkpt.pt
    # and every other tool in the repo takes the dir), so resolve it; anything
    # else raises.
    if checkpoint and os.path.isdir(checkpoint):
        cand = os.path.join(checkpoint, "chkpt.pt")
        if not os.path.isfile(cand):
            raise FileNotFoundError(
                f"{checkpoint} is a directory with no chkpt.pt in it.  Pass the "
                f"checkpoint dir or the .pt file; passing neither used to run "
                f"silently on the BASE weights.")
        checkpoint = cand
    elif checkpoint and not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"--checkpoint {checkpoint} does not exist.  This used to be "
            f"ignored, and the eval ran on the base weights with a healthy-"
            f"looking log.")

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # config_overrides FORCES graft-building flags that config.json does not
    # carry -- the only way to run the pre-launch gates on a PARENT checkpoint,
    # before any arm with that geometry exists (e.g. latent_carry on the E-only
    # heal checkpoint the Z arms branch from).
    #
    # It is printed, loudly, because a geometry the checkpoint was not trained
    # with is exactly the silent mismatch every other guard in this file exists
    # to prevent -- the difference is that here it is deliberate, and a
    # deliberate override that nothing announces is indistinguishable from the
    # accident.
    if config_overrides:
        for _k, _v in config_overrides.items():
            print(f"[cortex] OVERRIDE config.{_k}: "
                  f"{getattr(config, _k, '<absent>')!r} -> {_v!r}  "
                  f"(NOT what this checkpoint trained with unless they match)")
            setattr(config, _k, _v)
    if memory_slots is not None:
        # Force the graft on (model_name must use the grafted modeling file).
        config.use_memory = True
        config.memory_slots = memory_slots
    # accum_max is the prefix buffer's FIFO cap (buffers.py: out[:, -max_vecs:]),
    # a plain int with no parameter shape behind it -- unlike accum_vecs, which
    # sizes summary_emb and can never be varied after training.  Training sized
    # it to exactly cross_chunks * accum_vecs and train.py ASSERTS that, so the
    # trim branch never fires during training and always fires at eval on any
    # example longer than (accum_max / accum_vecs) chunks.  Overriding it here
    # is the only way to ask what the memory does when it is allowed to keep
    # the context it was given.
    if accum_max is not None:
        config.accum_max = int(accum_max)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=dtype,
    )

    # Optional overlay of finetuned weights from a train.py checkpoint.
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] overlaid {checkpoint}: {len(missing)} missing / "
              f"{len(unexpected)} unexpected keys")

    # Fail loud if memory was requested but the graft didn't load: the grafted
    # modeling file falls back to CortexMemory=None when `import cortex_graft`
    # fails (e.g. evals launched from outside the repo root), which would
    # SILENTLY run as a no-memory baseline despite use_memory=True.
    if getattr(config, "use_memory", False) and getattr(_unwrap(model), "cortex", None) is None:
        raise RuntimeError(
            "config.use_memory is set but model.cortex is None — the cortex_graft "
            "import failed (run evals from the repo root) or model_name is not a "
            "graft-prepared dir. Eval would silently run as a no-memory baseline."
        )

    model = model.to(device=device, dtype=dtype).eval()

    # Report the buffer geometry that actually got built, not the one that was
    # requested.  config.accum_max is absent from some prepared checkpoint dirs,
    # in which case the graft silently falls back to 128 -- the HEAL phase's
    # value, not the arm's 256 -- and halves the memory's horizon without
    # anything in the log saying so.
    buf = accumulating_buffer(model)
    if buf is not None:
        horizon = buf.max_vecs // max(buf.n_vec, 1)
        src = "OVERRIDE" if accum_max is not None else \
              ("config.json" if hasattr(config, "accum_max") else "CODE DEFAULT")
        print(f"[cortex] carry buffer: n_vec={buf.n_vec} max_vecs={buf.max_vecs} "
              f"({src}) -> holds the newest {horizon} chunks; older writes are "
              f"dropped by the FIFO")
        if src == "CODE DEFAULT":
            print(f"[cortex] WARNING: this checkpoint's config.json has no "
                  f"accum_max. Verify it against the training value before "
                  f"reporting anything from this run.")
    return model, config


@torch.no_grad()
def prime_cross_state(model, chunks, num_steps, passes_per_chunk=1):
    """Run priming chunks through the model, carrying M_cross across them.
    passes_per_chunk > 1 runs each chunk through the FULL model that many
    times (M_cross carried pass-to-pass), so the buffer gets multiple writes
    per chunk instead of one — the multi-pass fill the LM2 buffer design
    intends.  Returns the final buffer, or None for models without cross
    state (base / parcae-style) — those see only the final prediction chunk,
    which is exactly the no-memory control condition."""
    if not has_cross_state(model) or not chunks:
        return None
    device = next(model.parameters()).device
    m_cross = None
    for chunk in chunks:
        chunk = chunk.to(device)
        for _ in range(max(passes_per_chunk, 1)):
            out = model(input_ids=chunk, num_steps=num_steps,
                        m_cross_in=m_cross, return_m_cross=True)
            m_cross = out.get("m_cross")
    return m_cross


@torch.no_grad()
def ccot_prime(model, input_ids, num_steps, passes, m_cross_init=None):
    """Mixed CCoT: run `passes` full silent forward passes over the SAME
    tokens, feeding each pass's M_cross write into the next pass's read —
    latent multi-pass 'thinking' before any token is generated.  m_cross_init
    seeds the first pass (e.g. a buffer primed on earlier context chunks).
    Returns the final buffer (m_cross_init unchanged when the model has no
    cross state or passes <= 0)."""
    if passes <= 0 or not has_cross_state(model):
        return m_cross_init
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    m_cross = m_cross_init
    for _ in range(passes):
        out = model(input_ids=input_ids, num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True)
        m_cross = out.get("m_cross")
    return m_cross


def seed_example(example_id: str) -> None:
    """Pin the RNG from a stable example id.

    initialize_state draws s0 ~ trunc_normal (raven_modeling_minimal_olmo.py:982),
    so two runs of the same example land on different s0 draws.  For unpaired
    count comparisons that is just noise; for the PAIRED carry-on/carry-off
    contrast it is noise that does not cancel, because the two conditions are
    separate jobs.  Seeding from the example id immediately before the scored
    forward makes both conditions consume the same draw, so the only difference
    left between them is the carry -- which is the whole point of the design.

    Call it after priming (which consumes RNG in the carry-on condition and
    not in the carry-off one) and before generation/scoring.
    """
    torch.manual_seed(zlib.crc32(example_id.encode("utf-8")) & 0x7fffffff)


@torch.no_grad()
def score_continuation(model, prompt_ids, cont_ids, num_steps, m_cross=None,
                       eos_id=None):
    """Teacher-forced NLL of `cont_ids` given `prompt_ids`, in nats.

    The accuracy metric these evals report is 0/1 containment, whose variance
    is ~p(1-p) per example and which therefore cannot resolve a sub-point
    effect at any n we can afford (tools/power_longcontext.py).  The gold
    answer's NLL under the same forward is continuous, is measured on the same
    example in both conditions, and is the metric the carry ablation already
    resolves to ~0.001 nats at n=150.  One extra forward per example.

    Uses the same forward configuration as greedy_generate's prefill
    (prefix_write=False, prefix_read=True, explicit position_ids), so this
    scores the distribution the generator actually decoded from.

    Returns a dict: nll_sum, n_tok, and two diagnostics read off the SAME
    forward at the first generated position -- p_eos_first and entropy_first.
    Those two are the readout for the LongMemEval failure mode: the carry-on
    condition stops on EOS after ~10 words where carry-off runs to the cap, and
    containment scoring then loses because the gold string surfaces later.
    p_eos_first measures that directly instead of inferring it from lengths.
    entropy_first catches the other way a carry can go wrong -- a buffer far
    outside its trained regime flattening or collapsing the output
    distribution -- which is what "did it blow up" actually means here.
    """
    device = next(model.parameters()).device
    prompt_ids, cont_ids = prompt_ids.to(device), cont_ids.to(device)
    if cont_ids.numel() == 0:
        return {"nll_sum": float("nan"), "n_tok": 0}
    full = torch.cat([prompt_ids, cont_ids], dim=1)
    pos = torch.arange(full.shape[1], device=device).unsqueeze(0)
    out = model(input_ids=full, num_steps=num_steps, position_ids=pos,
                m_cross_in=m_cross, return_m_cross=False,
                prefix_write=False, prefix_read=True)
    # Logit at position i predicts token i+1, so the answer's own logits start
    # one step before the answer.
    first = prompt_ids.shape[1] - 1
    logits = out["logits"][:, first: -1, :].float()
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
    head = logp[0, 0]
    res = {"nll_sum": float(nll.sum().item()), "n_tok": int(cont_ids.numel()),
           "entropy_first": float(-(head.exp() * head).sum().item())}
    if eos_id is not None:
        res["p_eos_first"] = float(head[eos_id].exp().item())
    return res


@torch.no_grad()
def rank_candidates(model, tokenizer, prompt_ids, candidates, num_steps,
                    m_cross=None):
    """Score every candidate answer under one prompt.  Returns a list of
    {text, nll_sum, n_tok} in the order given.

    Why this exists.  Free generation + substring containment measures how much
    the model says at least as much as what it knows: the carry raises P(EOS),
    generations shorten, and the gold string stops appearing -- which is how
    LongMemEval inverted.  Ranking never asks the model to emit anything, so
    that entire failure mode drops out.  It also gives the eval a defined chance
    level (1/len(candidates)); the containment numbers sit BELOW uniform
    guessing on qa2/qa3, which is not a statement about the model.

    Scored sequentially rather than as one padded batch on purpose: the prefix
    splice, the m_cross batch dim and the recurrent init all have to agree about
    batch shape, and a padding bug there would be silent and would land on both
    conditions unequally.  Six forwards over a 512-token window is ~10% on top
    of priming at 32k, which is not worth that risk.
    """
    out = []
    for text in candidates:
        ids = tokenizer(" " + str(text).strip(),
                        add_special_tokens=False).input_ids
        cont = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        s = score_continuation(model, prompt_ids, cont, num_steps,
                               m_cross=m_cross)
        out.append({"text": text, "nll_sum": s["nll_sum"], "n_tok": s["n_tok"]})
    return out


def buffer_geometry(model):
    """(n_vec, max_vecs, chunks_held) for the carry buffer, or None.

    chunks_held is how many chunks of writes survive the FIFO -- the memory's
    horizon in chunks.  Recorded per example so the eviction regime is a column
    in the results rather than something reconstructed afterwards from the
    training config.
    """
    buf = accumulating_buffer(model)
    if buf is None:
        return None
    n_vec = max(int(buf.n_vec), 1)
    return n_vec, int(buf.max_vecs), int(buf.max_vecs) // n_vec


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, max_new_tokens, num_steps,
                    m_cross=None, stop_on_newline=False, use_cache=True,
                    stop_fn=None):
    """Greedy decoding.  An optional primed m_cross buffer is held fixed as
    read-only context for every step.  Returns the generated text.

    stop_fn(decoded_so_far) -> bool ends generation early; stop_on_newline is
    the common case kept as its own flag.  eval_gsm8k passes a stop_fn rather
    than keeping its own copy of this loop, so there is one decode path to get
    right instead of two.

    use_cache=True (default) decodes incrementally against a HuginnDynamicCache:
    prefill the prompt once, then forward ONE token per step.  The uncached path
    (use_cache=False) re-forwards the whole prefix every step — it is the
    original eval_gsm8k behaviour, kept as the reference implementation that
    tests/smoke_kv_cache.py checks the cached path against.  Both must produce
    identical text; the cached path is ~2 orders of magnitude cheaper.

    Prefix-memory models need two things for the cache to be CORRECT, not just
    fast (see cortex_graft.prefix_pack):
      * prefix_write=False everywhere — a cached single-token query runs with
        is_causal=False and would otherwise attend to the summary slots, which
        the uncached causal mask never allows;
      * prefix_read=False on incremental steps — the carry is already in the
        cache from the prefill, and re-splicing it would double-count it and
        shift every absolute position key the cache uses.

    NOT bit-identical to the uncached path, and it cannot be: initialize_state
    draws s0 ~ trunc_normal(std=sqrt(2/(5*n_embd))) with the shape of whatever
    it is handed (raven_modeling_minimal_olmo.py:982), so a forward over 1 token
    and a forward over the whole prefix consume different draws.  The two paths
    are distributionally equivalent samples of the same model, not replays of
    one computation.  tests/smoke_kv_cache.py pins the part that IS meant to be
    exact by passing init_scale=0.0, which makes s0 deterministically zero and
    isolates the cache/packing arithmetic from the noise.
    """
    device = next(model.parameters()).device
    generated = input_ids.to(device)
    prompt_len = generated.shape[1]
    eos_id = tokenizer.eos_token_id

    def _stop(tok) -> bool:
        if eos_id is not None and tok.item() == eos_id:
            return True
        if not (stop_on_newline or stop_fn):
            return False
        new = tokenizer.decode(generated[0, prompt_len:])
        if stop_on_newline and "\n" in new:
            return True
        return bool(stop_fn(new)) if stop_fn else False

    if not use_cache:
        for _ in range(max_new_tokens):
            out = model(input_ids=generated, num_steps=num_steps,
                        m_cross_in=m_cross, return_m_cross=False,
                        prefix_write=False)
            next_tok = out["logits"][0, -1].argmax(dim=-1).view(1, 1)
            generated = torch.cat([generated, next_tok], dim=1)
            if _stop(next_tok):
                break
        return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

    # Prefill: whole prompt, carry spliced in, no summary slots.
    pos = torch.arange(prompt_len, device=device).unsqueeze(0)
    out = model(input_ids=generated, num_steps=num_steps, position_ids=pos,
                m_cross_in=m_cross, return_m_cross=False,
                use_cache=True, prefix_write=False, prefix_read=True)
    cache = out.past_key_values
    next_tok = out["logits"][0, -1].argmax(dim=-1).view(1, 1)
    generated = torch.cat([generated, next_tok], dim=1)

    for i in range(max_new_tokens - 1):
        if _stop(next_tok):
            break
        # Absolute index of the token being fed; prefix_pack re-applies the
        # same +1 shift the prefill used, so the numbering is continuous.
        pos = torch.tensor([[prompt_len + i]], device=device)
        out = model(input_ids=next_tok, num_steps=num_steps, position_ids=pos,
                    m_cross_in=m_cross, return_m_cross=False,
                    past_key_values=cache, use_cache=True,
                    prefix_write=False, prefix_read=False)
        cache = out.past_key_values
        next_tok = out["logits"][0, -1].argmax(dim=-1).view(1, 1)
        generated = torch.cat([generated, next_tok], dim=1)

    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
