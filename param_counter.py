def warmup_hist(this_checkpoint, warmup_duration, max_steps, max_rec, warmup_type="linear"):
    if warmup_type == "linear":
        # f(t) = t  => f^{-1}(y) = y
        finv = lambda y: y
    elif warmup_type == "1-sqrt":
        # f(t) = 1 - sqrt(1 - t)  => f^{-1}(y) = 1 - (1 - y)^2
        finv = lambda y: 1.0 - (1.0 - y) ** 2
    else:
        raise ValueError(f"Unsupported warmup_type: {warmup_type}")

    warmup_steps = int(warmup_duration * max_steps)
    S = max(0, min(this_checkpoint, max_steps))          # steps completed so far (cap at max_steps)
    S_warm = min(S, warmup_steps)

    counts = {r: 0 for r in range(1, max_rec + 1)}

    # Add warmup contribution for each integer recurrence r
    for r in range(1, max_rec + 1):
        # start = ceil_div(finv(r - 1) * warmup_steps, max_rec)   # first step index with recurrence r
        # end_excl = ceil_div(finv(r) * warmup_steps, max_rec)      # first step index after r
        start = int(finv((r - 1) / max_rec) * warmup_steps)
        end_excl = int(finv(r / max_rec) * warmup_steps)
        # overlap with [0, S_warm)
        overlap = max(0, min(end_excl, S_warm) - start)
        counts[r] += overlap

    # Add post-warmup contribution (stays at max_rec)
    if S > warmup_steps:
        counts[max_rec] += S - warmup_steps

    # prune zeros if you like:
    return {r: c for r, c in counts.items() if c > 0}

def count_params(model):
    param_counts = {
        "embeddings + lm_head": 0,
        "prelude": 0,
        "rec_block": 0,
        "coda": 0,
        "total_not_emb_or_lm_head": 0,
        "total": 0,
    }

    for name, param in model.named_parameters():
        num_params = param.numel()
        param_counts["total"] += num_params

        # Embeddings + lm_head
        if name.startswith("transformer.wte") or name.startswith("lm_head"):
            param_counts["embeddings + lm_head"] += num_params
            continue

        # Prelude
        if name.startswith("transformer.prelude"):
            param_counts["prelude"] += num_params
            param_counts["total_not_emb_or_lm_head"] += num_params
            continue

        # Core blocks (rec_block)
        if name.startswith("transformer.core_block"):
            param_counts["rec_block"] += num_params
            param_counts["total_not_emb_or_lm_head"] += num_params
            continue

        # Coda
        if name.startswith("transformer.coda"):
            param_counts["coda"] += num_params
            param_counts["total_not_emb_or_lm_head"] += num_params
            continue

        # Everything else (e.g., adapter, ln_f, etc.)
        param_counts["total_not_emb_or_lm_head"] += num_params

    return param_counts

def count_params_with_rec(param_counts, num_rec, num_grad_rec=8):
    ## FLOPs Calc
    n = max(0, num_rec - num_grad_rec) # no grad
    k = min(num_rec, num_grad_rec) # grad
    prams_with_grad = param_counts["prelude"] + (param_counts["rec_block"] * k) + param_counts["coda"]
    prams_no_grad = param_counts["rec_block"] * n
    param_counts["flops_times_by_6d"] = prams_with_grad + ((1/3) * prams_no_grad)
    # 6 * D * N_1 + 2 * D * N_2 where N_1 = model params not including the recurrences with no grad and N_2 = model params from recurrences with no grad (i.e. N_1 + N_2 = effective params = N) so Flops = 2 * D  * (3* N_1 + N_2)

    param_counts["rec_block"] = param_counts["rec_block"] * num_rec
    param_counts["total_not_emb_or_lm_head"] = param_counts["prelude"] + param_counts["rec_block"] + param_counts["coda"]
    param_counts["total"] = param_counts["total_not_emb_or_lm_head"] + param_counts["embeddings + lm_head"]
    return param_counts
