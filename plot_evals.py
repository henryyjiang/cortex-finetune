import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from param_counter import count_params, count_params_with_rec, warmup_hist
from transformers import AutoModelForCausalLM
import copy
from matplotlib import font_manager
from natsort import natsorted
import numpy as np

def import_times_new_roman(this_font_manager=font_manager, this_plt=plt, font_size=16):
    this_font_manager.fontManager.addfont(f"paper_plots/Times New Roman.ttf")
    this_plt.rcParams["font.family"] = "Times New Roman"
    this_plt.rcParams["font.size"] = font_size


task_to_key = {
    "arc_challenge": "acc_norm,none", 
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "lambada_openai": "acc,none",
    "mmlu": "acc,none",
    "openbookqa": "acc_norm,none",
    "piqa": "acc_norm,none",
    "social_iqa": "acc,none",
    "winogrande": "acc,none",

    "minerva_math": "math_verify,none",
    "gsm8k_cot_sean": "exact_match,flexible-extract",
}

task_to_random = {
    "arc_challenge": 0.25, 
    "arc_easy": 0.25,
    "hellaswag": 0.25,
    "lambada_openai": 0.25,
    "mmlu": 0.25,
    "openbookqa": 0.25,
    "open_openbookqa": 0.25,
    "piqa": 0.5,
    "social_iqa": 0.25,
    "winogrande": 0.5,

    "minerva_math": 0.0,
    "gsm8k_cot_sean": 0.0,
}

CACHE_FILE = "model_size_cache.json"
def _load_model_size_cache(path: str = CACHE_FILE) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            raw = json.load(f)
        # JSON makes keys strings; convert nested keys back to int
        return {m: {int(k): v for k, v in sizes.items()} for m, sizes in raw.items()}
    return {}

def _save_model_size_cache(cache: dict, path: str = CACHE_FILE):
    with open(path, "w") as f:
        json.dump(cache, f)

model_size_cache = _load_model_size_cache()

def get_flops(all_counts, train_rec, this_chkpt, max_steps, batch_size=2**20, warmup_duration=0.25, warmup_type="linear"): # 1M batch size
    if train_rec == 1:
        return all_counts[train_rec]["flops_times_by_6d"] * 6 * this_chkpt * batch_size
    else:
        warmups = warmup_hist(this_chkpt, warmup_duration, max_steps=max_steps, max_rec=train_rec, warmup_type=warmup_type)
        flops_times_checkpoints = sum(all_counts[k]["flops_times_by_6d"] * v for k, v in warmups.items())
        return flops_times_checkpoints * 6 * batch_size
    
def make_df(runs):
    global model_size_cache

    all_data = []
    for dir, short_name, extra_dict in runs:
        chkpt_dirs = os.listdir(dir)
        for chkpt in chkpt_dirs:
            json_files_recursive = glob.glob(f"{dir}/{chkpt}/**/*.json", recursive=True)
            for json_file in json_files_recursive:
                with open(json_file, "r") as f:
                    data = json.load(f)

                model_path = data["model_name"]

                dict_name = "/".join(model_path.split("/")[:-1])
                if dict_name not in model_size_cache.keys():
                    print(f"Adding {dict_name} to model size cache")
                    if ("TinyLlama-1.1B-intermediate-step-1431k-3T" in dir) or ("Llama-3.2-1B-untied" in dir) or ("OLMo-2-0425-1B-step1907359" in dir):
                        if model_path.startswith("models/"):
                            model_path = model_path.replace("models/", "")

                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        attn_implementation="sdpa",
                        trust_remote_code=True,
                    )
                    if ("TinyLlama-1.1B-intermediate-step-1431k-3T" in dir) or ("non_recur" in dir) or ("Llama-3.2-1B-untied" in dir) or ("OLMo-2-0425-1B-step1907359" in dir):
                        total_params = sum(
                            p.numel() for name, p in model.named_parameters()
                            if not (("embed_tokens" in name) or ("lm_head" in name))
                        )
                        model_size_cache[dict_name] = {1: {"total_not_emb_or_lm_head": total_params, "flops_times_by_6d": total_params}}
                    else:
                        counts = count_params(model)
                        size_dict = {k: count_params_with_rec(copy.deepcopy(counts), num_rec=k) for k in range(1,33)}
                        model_size_cache[dict_name] = size_dict

                if ("TinyLlama-1.1B-intermediate-step-1431k-3T" in dir) or ("Llama-3.2-1B-untied" in dir) or ("OLMo-2-0425-1B-step1907359" in dir):
                    num_rec = 1
                    this_chkpt = 0
                elif "non_recur" in dir:
                    # print(json_file)
                    num_rec = 1
                    this_chkpt = int(chkpt.replace("model_only_chkpt_",""))
                else:
                    # print(json_file)
                    try:
                        num_rec = data["configs"]["hellaswag"]["metadata"]["mean_recurrence"]
                    except:
                        if "minerva_math" in data["configs"].keys():
                            num_rec = data["configs"]["minerva_math"]["metadata"]["mean_recurrence"]
                        elif "gsm8k_cot_sean" in data["configs"].keys():
                            num_rec = data["configs"]["gsm8k_cot_sean"]["metadata"]["mean_recurrence"]
                        else:
                            print(json_file)
                            print(f"didn't find a matching task key in {data['configs'].keys()}")
                            print()
                            exit()
                    this_chkpt = int(chkpt.replace("model_only_chkpt_",""))
                # print(data["results"])
                # exit()
                for k, v in data["results"].items():
                    if k in task_to_key.keys():
                        task_key = task_to_key[k]
                        all_data.append(
                            {
                                "model": short_name,
                                "chkpt": this_chkpt,
                                "task": k,
                                "acc": v[task_key],
                                "stderr": v[task_key.replace(",","_stderr,")],
                                "train_rec": extra_dict["train_rec"],
                                "num_rec": num_rec,
                                "cooldown": "cooldown" in short_name,
                                "effective_params": model_size_cache[dict_name][num_rec]["total_not_emb_or_lm_head"],
                                "FLOPs": get_flops(model_size_cache[dict_name], extra_dict["train_rec"], this_chkpt,  max_steps=extra_dict.get("max_steps", 25_000), warmup_duration=extra_dict.get("warmup_duration", 0.25), warmup_type=extra_dict.get("warmup_type", "linear"))
                            }
                        )
                        # print(short_name, extra_dict.get("warmup_type", "linear"))
    _save_model_size_cache(model_size_cache)
    return pd.DataFrame(all_data)

def plot(runs, save_prefix, save_df=False):
    df = make_df(runs)
    if save_df:
        df.to_json(f"paper_plots/data/{save_prefix}.jsonl", orient="records", lines=True)
    print(df)

    num_rec_styles = {
        1: '-',
        2: '--',
        4: '-.',
        8: ':',
        16: (0, (3, 1, 1, 1)),  # dash-dot-dot
        32: (0, (1, 1)),        # densely dotted
    }

    for task_name in df["task"].unique():
        x_axes = ["chkpt"]
        if task_name in ["minerva_math", "gsm8k_cot_sean"]:
            x_axes = ["chkpt", "FLOPs", "effective_params"]
        for x_axis in x_axes:
            subset = df[df['task'] == task_name]
            print(f"{task_name}: {subset['model'].unique()}")

            plt.figure(figsize=(8, 4))
            plt.grid()

            # Assign consistent color per model
            models = sorted(subset["model"].unique())
            # model_colors = dict(zip(models, plt.cm.tab10.colors))  # up to 10 distinct colors
            cmap = plt.get_cmap('tab20')  # 20 distinct colors; cycles if >20
            model_colors = {m: cmap(i % cmap.N) for i, m in enumerate(models)}

            for model_name in models:
                model_subset = subset[subset["model"] == model_name]
                for num_rec in sorted(model_subset["num_rec"].unique()):
                    style = num_rec_styles.get(num_rec, '-')  # fallback to solid
                    data = model_subset[model_subset["num_rec"] == num_rec].sort_values(x_axis)

                    plt.errorbar(
                        data[x_axis],
                        data["acc"],
                        label=f"{model_name} (num_rec={num_rec})",
                        color=model_colors[model_name],
                        marker='o',
                        capsize=4,
                        linestyle=style
                    )

            plt.axhline(y=task_to_random[task_name], color='black', linestyle='--', label='Random Baseline')

            plt.xlabel("Step" if x_axis == "chkpt" else x_axis)
            plt.ylabel("Accuracy")

            # Legend for models (color)
            model_legend_handles = [
                matplotlib.lines.Line2D([0], [0], color=model_colors[model], lw=2, label=model)
                for model in models
            ]

            # Legend for num_rec (linestyle)
            num_rec_legend_handles = [
                matplotlib.lines.Line2D([0], [0], color='black', lw=2, linestyle=style, label=f"num_rec={num_rec}")
                for num_rec, style in num_rec_styles.items()
            ]

            second_legend = plt.legend(
                handles=num_rec_legend_handles,
                title="Num Rec (Line Style)",
                loc='upper center',
                bbox_to_anchor=(0.75, -0.1),
                ncol=2,
                frameon=False
            )
            plt.gca().add_artist(second_legend)

            first_legend = plt.legend(
                handles=model_legend_handles,
                title="Model (Color)",
                loc='upper center',
                bbox_to_anchor=(0.2, -0.1),
                ncol=2,
                frameon=False
            )
            # plt.gca().add_artist(first_legend)

            axis_name = f"_{x_axis}" if x_axis != "chkpt" else ""
            plt.savefig(f"eval_plots_2/{save_prefix}_{task_name}{axis_name}.pdf", bbox_inches="tight")
            plt.clf()
            print(f"eval_plots_2/{save_prefix}_{task_name}{axis_name}.pdf")

if __name__ == "__main__":
    # This is an example of how I would plot my olmo runs :)
    runs = [
        ("eval_outputs/huginn_llama/16_node_bf16_mixed_mbs_8_bs_16_olmo_2_0425_1b_step1907359_4_6_4_last_6_layer_5e_5_schedule_n_75_cooldown_9_muon_001_nemotron_max_mean_rec_32_sqrt_50k_steps_attempt_2", "4,6,4 (Train Recurrence=32)", {"train_rec": 32, "max_steps": 50_000, "warmup_duration": 0.75, "warmup_type":"1-sqrt"}),
        ("eval_outputs/huginn_llama/16_node_bf16_mixed_mbs_8_bs_16_olmo_2_0425_1b_step1907359_4_6_4_last_6_layer_5e_5_schedule_n_75_cooldown_9_muon_001_nemotron_max_mean_rec_16_sqrt_50k_steps_attempt_2", "4,6,4 (Train Recurrence=16)", {"train_rec": 16, "max_steps": 50_000, "warmup_duration": 0.75, "warmup_type":"1-sqrt"}),
        ("eval_outputs/huginn_llama/16_node_bf16_mixed_mbs_8_bs_16_olmo_2_0425_1b_step1907359_4_6_4_last_6_layer_5e_5_schedule_n_75_cooldown_9_muon_001_nemotron_max_mean_rec_8_sqrt_50k_steps_attempt_2", "4,6,4 (Train Recurrence=8)", {"train_rec": 8, "max_steps": 50_000, "warmup_duration": 0.75, "warmup_type":"1-sqrt"}),
        ("eval_outputs/huginn_llama/16_node_bf16_mixed_mbs_8_bs_16_olmo_2_0425_1b_step1907359_4_6_4_last_6_layer_5e_5_schedule_n_75_cooldown_9_muon_001_nemotron_max_mean_rec_4_sqrt_50k_steps_attempt_2", "4,6,4 (Train Recurrence=4)", {"train_rec": 4, "max_steps": 50_000, "warmup_duration": 0.75, "warmup_type":"1-sqrt"}),
        ("eval_outputs/huginn_llama/16_node_bf16_mixed_mbs_8_bs_16_non_recur_olmo_2_1907359_5e_5_no_compile_cooldown_9_muon_001_nemotron_50k_steps_attempt_2", "Olmo-2-step1907359\nStatic Depth", {"train_rec": 1}),
        ("eval_outputs/OLMo-2-0425-1B-step1907359", "OLMo-2-0425-1B-step1907359 hf", {"train_rec": 1})
    ]
    save_prefix = "olmo"

    plot(runs, save_prefix, save_df=False)