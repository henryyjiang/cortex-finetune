import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import os
import json
from itertools import cycle
from matplotlib import font_manager
import matplotlib
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM
import copy
import seaborn as sns
from natsort import natsorted

matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(
    color=sns.color_palette("colorblind")
)

parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(parent))
from param_counter import count_params, count_params_with_rec, warmup_hist
from plot_evals import task_to_random

def import_times_new_roman(this_font_manager=font_manager, this_plt=plt, font_size=16):
    this_font_manager.fontManager.addfont(f"Times New Roman.ttf")
    this_plt.rcParams["font.family"] = "Times New Roman"
    this_plt.rcParams["font.size"] = font_size

def generic_loss_plotter(save_name, df, run_id_to_name, legend_under_plot=False):
    import_times_new_roman(font_size=24)

    unique_names = list(set(run_id_to_name.values()))
    colors = sns.color_palette("colorblind", len(unique_names)) #plt.get_cmap("tab10", len(unique_names))  # distinct color map
    name_to_color = {name: colors[i] for i, name in enumerate(unique_names)}

    # Plotting log-log plot for each run_id
    plt.figure(figsize=(8, 4))

    already_used = set()
    for run_id, name in run_id_to_name.items():
        subset = df[df['run_id'] == run_id]
        if name in already_used:
            plt.loglog(subset['step'], subset['loss'], linestyle='-', color=name_to_color[name])
        else:
            plt.loglog(subset['step'], subset['loss'], linestyle='-', label=name, color=name_to_color[name])
            already_used.add(name)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))
    ax.yaxis.set_minor_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))

    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    if legend_under_plot:
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=3
        )
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"{save_name}.pdf", bbox_inches="tight")

def shortgpt(df):
    run_id_to_name = {
        "kzjs4sa7": "4,8,4 last 8 layers",# 8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_muon_001_nemotron
        "q7avsp76": "4,8,4 last 8 layers",# 8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_muon_001_nemotron_part_2
        "481mdoxd": "4,8,4 ShortGPT layers",# 8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_shortgpt_layers_5e_5_schedule_n_25_cooldown_9_muon_001_nemotron
        "48nd83l1": "4,8,4 ShortGPT layers",# 8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_shortgpt_layers_5e_5_schedule_n_25_cooldown_9_muon_001_nemotron_part_2
        # "bvwhypkq": "Tinyllama drop 6 layers Angular",# 8_node_bf16_mixed_mbs_8_bs_32_non_recur_tinyllama_remove_6_layers_shortgpt_angular_5e_5_no_compile_cooldown_9_muon_001_nemotron
        # "h9g0eb50": "Tinyllama drop 14 layers Angular",# 8_node_bf16_mixed_mbs_8_bs_32_non_recur_tinyllama_remove_14_layers_shortgpt_angular_5e_5_no_compile_cooldown_9_muon_001_nemotron
        "j76tvdtu": "Tinyllama Depth 8 ShortGPT",# 8_node_bf16_mixed_mbs_8_bs_32_non_recur_tinyllama_remove_14_layers_shortgpt_5e_5_no_compile_cooldown_9_muon_001_nemotron
        "ttn21chb": "Tinyllama Depth 16 ShortGPT",# 8_node_bf16_mixed_mbs_8_bs_32_non_recur_tinyllama_remove_6_layers_shortgpt_5e_5_no_compile_cooldown_9_muon_001_nemotron
    }
    # could show this plot in terms of gsm8k perf too
    generic_loss_plotter("shortgpt_loss", df, run_id_to_name)

def which_layers(df):
    run_id_to_name = {
        '3oeusxzc': "[0,1],[10,11,12,13],[14,15]", # huginn_llama_1b_last_4_layers_5e_5_4_gpu_2M_samples_0_cooldown
        'obwca535': "[0,1],[2,5,8,11],[14,15]", # huginn_llama_1b_evenly_spaced_layers_start_2_5e_5_4_gpu_2M_samples_0_cooldown
        'ofjzvvi0': "[0,1],[4,7,10,13],[14,15]", # huginn_llama_1b_evenly_spaced_layers_start_4_5e_5_4_gpu_2M_samples_0_cooldown
        # 'or45gmqm': "[0,1],[2,3,4,5],[14,15] compiled", # huginn_llama_1b_5e_5_4_gpu_3M_samples_0_cooldown_compile_better_metrics
        'r14701tv': "2,4,2 Takase", # llama_1b_init_from_scratch_jonas_init_5e_5_4_gpu_2M_samples_0_cooldown_better_metrics
        'zugjr1e9': "[0,1],[2,3,4,5],[14,15]", # huginn_llama_1b_5e_5_4_gpu_2M_samples_0_cooldown_better_metrics
    }
    generic_loss_plotter("which_layers", df, run_id_to_name)

def long_runs(save_name, df, run_id_to_name):
    # REC_MIN, REC_MAX = 1, 32
    rec_levels = [1, 2, 4, 8, 16, 32]
    rec_to_idx = {r: i for i, r in enumerate(rec_levels)}
    REC_MIN_IDX, REC_MAX_IDX = 0, len(rec_levels) - 1

    def darken(color, num_rec, max_darken=0.7, max_lighten=0.6):
        """
        Darken a base color towards black by a factor in [0, 1].
        t is the normalized recurrence level.
        max_darken limits how dark we go (0=no change, 1=black).
        """
        # t = (num_rec - REC_MIN) / (REC_MAX - REC_MIN)

        t = (rec_to_idx[num_rec] - REC_MIN_IDX) / (REC_MAX_IDX - REC_MIN_IDX)
        # r, g, b = matplotlib.colors.to_rgb(color)
        # k = 1.0 - max_darken * t
        # return (r * k, g * k, b * k)
    
        r, g, b = matplotlib.colors.to_rgb(color)

        centered_t = (t - 0.5) * 2  # now in [-1, +1]

        if centered_t < 0:  # lighten
            k = 1 - max_lighten * abs(centered_t)
            r, g, b = r + (1-r)*(1-k), g + (1-g)*(1-k), b + (1-b)*(1-k)
        else:               # darken
            k = 1 - max_darken * centered_t
            r, g, b = r * k, g * k, b * k

        return (r, g, b)

    num_rec_markers = {
        1: "o",   # circle
        2: "s",   # square
        4: "D",   # diamond
        8:  "X",  # x-filled
        16: "^",  # triangle up
        32: "P",  # plus-filled
    }
    eval_df = pd.read_json("data/tinyllama_long_run_table.jsonl", lines=True, engine="pyarrow")
    eval_df = eval_df[eval_df["task"] == "hellaswag"]

    import_times_new_roman(font_size=24)

    unique_names = list(set(run_id_to_name.values()))
    colors = sns.color_palette("colorblind", len(unique_names)) #plt.get_cmap("tab10", len(unique_names))  # distinct color map
    name_to_color = {name: colors[i] for i, name in enumerate(unique_names)}

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    already_used = set()
    for run_id, name in run_id_to_name.items():
        subset = df[df['run_id'] == run_id]
        eval_subset = eval_df[eval_df['model'] == name]

        # Left: Loss plot (log-log)
        if name in already_used:
            axes[0].loglog(subset['step'], subset['loss'], linestyle='-', color=name_to_color[name], linewidth=2)
        else:
            axes[0].loglog(subset['step'], subset['loss'], linestyle='-', label=name, color=name_to_color[name], linewidth=2)

        # Right: Accuracy plot (linear)
        for num_rec in eval_subset["num_rec"].unique():
            this_eval_subset = eval_subset[eval_subset["num_rec"] == num_rec].sort_values("chkpt")
            shade = darken(name_to_color[name], num_rec)
            # axes[1].plot(this_eval_subset['chkpt'], this_eval_subset['acc'], linestyle='-', color=shade, marker=num_rec_markers[num_rec])
            axes[1].plot(this_eval_subset['chkpt'], this_eval_subset['acc'], linestyle='-', color=shade, linewidth=3)


        already_used.add(name)

    # Formatting left plot (loss)
    axes[0].set_xlabel('Train Step')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, which="both", ls="--")
    # no x10^0s
    axes[0].yaxis.set_major_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))
    axes[0].yaxis.set_minor_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False)) 

    # Formatting right plot (accuracy)
    axes[1].set_xlabel('Train Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, ls="--")

    # Shared legend
    leg = fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.3, 1.1),
        ncol=2,
        # title="Model",
        frameon=False
    )
    for legline in leg.get_lines():
        legline.set_linewidth(3) 
    # marker_handles = [
    #     mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=10, label=f"{num_rec}")
    #     for num_rec, marker in num_rec_markers.items()
    # ]
    # fig.legend(
    #     handles=marker_handles,
    #     loc='upper center',
    #     bbox_to_anchor=(0.775, 0.05),
    #     ncol=3,
    #     title="Test Rec"
    # )
    rec_handles = []
    base_for_legend = (0.2, 0.2, 0.2)
    for r in rec_levels:
        shade = darken(base_for_legend, r)
        rec_handles.append(
            mlines.Line2D([], [], color=shade, linestyle='-', linewidth=4, label=f"{r}")
        )
    fig.legend(
        # handles=rec_handles,
        handles=[],
        loc='upper center',
        bbox_to_anchor=(0.79, 1.1),
        ncol=6,
        title="Test Recurrence (darker = larger)",
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(f"{save_name}.pdf", bbox_inches="tight")
    
def plot_1_extended_lines(df):
    import_times_new_roman(font_size=24)

    unique_run_ids = ["jrre5mv2", "re7ssigz"]
    # run_id_to_name = {"jrre5mv2": "Takase", "re7ssigz": "Huginn Llama"}
    run_id_to_name = {"re7ssigz": "Llama", "jrre5mv2": "Random"}
    # 54o491fi 64_node_bf16_mixed_mbs_8_bs_16_llama_1b_init_from_scratch_jonas_init_5e_5_250b_packed_wrapped_attempt_3_part_1
    # j3vpjyt9 64_node_bf16_mixed_mbs_8_bs_16_huginn_llama_1b_2_4_2_last_4_layers_5e_5_250b_packed_wrapped_attempt_3_part_1_cooldown_120b_132b
    # jrre5mv2 64_node_bf16_mixed_mbs_8_bs_16_llama_1b_init_from_scratch_with_monkeypatch_jonas_init_5e_5_250b_packed_wrapped_attempt_3_part_1
    # pi5tdel9 64_node_bf16_mixed_mbs_8_bs_16_llama_1b_init_from_scratch_jonas_init_5e_5_250b_packed_wrapped_attempt_3_part_1_cooldown_120b_132b
    # re7ssigz 64_node_bf16_mixed_mbs_8_bs_16_huginn_llama_1b_2_4_2_last_4_layers_5e_5_250b_packed_wrapped_attempt_3_part_1
    
    # generic_loss_plotter("long_runs", df, run_id_to_name, legend_under_plot=True)
    long_runs("long_runs", df, run_id_to_name)
    generic_loss_plotter("long_runs_inc_cooldown", df, run_id_to_name | {"pi5tdel9": "Random Cooldown", "j3vpjyt9": "Llama Cooldown"})
    generic_loss_plotter("long_runs_mp", df, {"54o491fi": "Our emb_scale", "jrre5mv2": "Geiping et al. emb_scale"})

    pivot_df = df.pivot(index="step", columns="run_id", values="loss")
    df = pivot_df.reset_index()
    df = df[unique_run_ids+["step"]]
    # df = df.dropna(subset=unique_run_ids, how='all')
    last_valid_index = df[['jrre5mv2', 're7ssigz']].dropna().index[-1]
    df = df.loc[:last_valid_index].reset_index(drop=True)

    from scipy.ndimage import uniform_filter1d
    from scipy.stats import linregress

    last_5k = df[df['step'] >= df['step'].max() - 5000]

    window_size = 10 
    smooth_loss1_5k = uniform_filter1d(last_5k['re7ssigz'], size=window_size)
    smooth_loss2_5k = uniform_filter1d(last_5k["jrre5mv2"], size=window_size)

    # Log transformation of smoothed data
    log_step_5k = np.log10(last_5k['step'])
    log_smooth_loss1_5k = np.log10(smooth_loss1_5k)
    log_smooth_loss2_5k = np.log10(smooth_loss2_5k)
    print(log_smooth_loss1_5k)
    print(log_smooth_loss2_5k)

    # Linear regression on smoothed data
    slope1_5k, intercept1_5k, _, _, _ = linregress(log_step_5k, log_smooth_loss1_5k)
    slope2_5k, intercept2_5k, _, _, _ = linregress(log_step_5k, log_smooth_loss2_5k)

    log_cross_x_5k = (intercept2_5k - intercept1_5k) / (slope1_5k - slope2_5k)
    cross_x_5k = 10 ** log_cross_x_5k
    cross_y_5k = 10 ** (slope1_5k * log_cross_x_5k + intercept1_5k)
    print(cross_x_5k, cross_y_5k)

    extended_steps = np.linspace(last_5k['step'].min(), cross_x_5k, 200)
    extended_loss1 = 10 ** (slope1_5k * np.log10(extended_steps) + intercept1_5k)
    extended_loss2 = 10 ** (slope2_5k * np.log10(extended_steps) + intercept2_5k)

    # Plotting log-log plot for each run_id
    plt.figure(figsize=(8, 4))
    lw=3.5
    for run_id in unique_run_ids:
        plt.loglog(df['step'], df[run_id], linestyle='-', label=f'{run_id_to_name[run_id]}', linewidth=lw)

    plt.loglog(extended_steps, extended_loss1, '--', label='Llama Smoothed Fit (last 5k)', linewidth=lw)
    plt.loglog(extended_steps, extended_loss2, '--', label='Random Smoothed Fit (last 5k)', linewidth=lw)
    toks_5k = cross_x_5k * 16 * 1024 * 64 * 4 # batch size x seq len x num nodes x num cards
    plt.scatter(cross_x_5k, cross_y_5k, color='red', zorder=5, label=f'Cross at Step={cross_x_5k:.1f}, Loss={cross_y_5k:.2f}\nToks={toks_5k:,.0f}', marker='x', s=100, linewidths=lw)

    # no x10^0s
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))
    ax.yaxis.set_minor_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False)) 


    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 0.95),
        frameon=False
    )
    plt.grid(True, which="both", ls="--")
    plt.savefig("long_runs_extended_lines.pdf", bbox_inches="tight")

# # def long_runs_compare_monkeypatch(df):

def muon(df):
    mask = (df["run_id"] == "55v46lal") & (df["step"] > 25000) # this was the run I put on for too long so ran with steps past 25k with lr 0
    df = df.loc[~mask]

    import_times_new_roman(font_size=24)
    # Get unique run_ids
    unique_run_ids = df['run_id'].unique()
    runs = []
    runs.append(['1i21hmi9', '55v46lal', 'c64qfy4d', 'n8n0rf4j', "4t43gsq2", "ii8fbwcd"])#, "hzdtwjdm", "l5whyuop", "djbkella"])
    runs.append(['96utpnl9','sgdx9ouy'])

    for idx, unique_run_ids in enumerate(runs):
        # unique_pairs = df[['run_id', 'run_name']].drop_duplicates()
        # unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))
        run_id_to_name = {
            '1i21hmi9': "4,8,4 Muon",#'8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_muon_001',
            '55v46lal': "4,8,4 Muon",#'8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_muon_001_part_2', 
            '96utpnl9': "Tinyllama Muon",#'8_node_bf16_mixed_mbs_8_bs_32_non_recur_tinyllama_5e_5_no_compile_cooldown_9_muon_001_data_fix',
            'c64qfy4d': "4,8,4 AdamW",#'8_node_bf16_mixed_mbs_8_bs_64_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_part_2',
            'hzdtwjdm': "4,8,4 AdamW 3e-5",#'8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_3e_5_schedule_n_25_cooldown_9',
            'n8n0rf4j': "4,8,4 AdamW",#'8_node_bf16_mixed_mbs_8_bs_64_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_part_2_actual',
            'sgdx9ouy': "Tinyllama AdamW",#'8_node_bf16_mixed_mbs_8_bs_32_non_recur_tinyllama_5e_5_no_compile_cooldown_9_data_fix'
            '4t43gsq2': "4,8,4 AdamW*", # 8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_ellisadam
            'ii8fbwcd': "4,8,4 AdamW*", # 8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_ellisadam
            'l5whyuop': "4,8,4 Muon wd", # 8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_muon_001_wd_1e_4
            'djbkella': "4,8,4 Muon wd", # 8_node_bf16_mixed_mbs_8_bs_32_tinyllama_4_8_4_last_8_layers_5e_5_schedule_n_25_cooldown_9_muon_001_wd_1e_4
        }

        unique_names = natsorted(list(set(run_id_to_name.values())))
        color_map  = sns.color_palette("colorblind", len(unique_names)) #plt.get_cmap("tab10", len(unique_names)) 
        name_to_color = {name: color_map[i] for i, name in enumerate(unique_names)}

        # Plotting log-log plot for each run_id
        plt.figure(figsize=(8, 4))
        if idx == 0:
            x_min, x_max = 100, df["step"].max()+1000
            y_min, y_max = 2, 7
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

        lw = 3
        seen_labels = []
        df['loss'] = df['loss'].fillna(1e20) # loss spike for NaN's
        for run_id in unique_run_ids:
            subset = df[df['run_id'] == run_id]

            label = run_id_to_name[run_id]
            x = subset['step'].values
            y = subset['loss'].values

            if idx == 1:
                window = 50 
                y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
                x_smooth = x[window-1:] 
                plot_x, plot_y = x_smooth, y_smooth
            else:
                plot_x, plot_y = x, y

            if label in seen_labels:
                plt.loglog(plot_x, plot_y, linestyle='-', color=name_to_color[label], linewidth=lw)
            else:
                plt.loglog(plot_x, plot_y, linestyle='-', label=label, color=name_to_color[label], linewidth=lw)
                seen_labels.append(label)

        if idx == 1:
            plt.yticks(np.arange(2.0, 2.3, 0.05))
            # plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        plt.xlabel('Train Step')
        if idx == 1:
            plt.ylabel('Smoothed Loss')
        else:
            plt.ylabel('Loss')
        
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))
        ax.yaxis.set_minor_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))
        if idx == 1:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter('%.2f'))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        plt.grid(True, which="both", ls="--")
        plt.savefig(f"muon_vs_adam._{idx}.pdf", bbox_inches="tight")
        plt.close()

def multi_recurrence_eval_df(wandb_models):
    models = os.listdir("../loss_over_rec_eval")
    models = list(set(wandb_models) & set(models))

    df = []
    for model in models:
        if (".pdf" in model) or ("old" in model):
            continue
        checkpoints = os.listdir(f"../loss_over_rec_eval/{model}")

        for checkpoint in checkpoints:
            with open(f"../loss_over_rec_eval/{model}/{checkpoint}", 'r') as f:
                data = json.load(f)

            mean_data = {int(k): float(np.mean(v)) for k,v in data.items()}
            df.append(
                {
                    "name": model,
                    "checkpoint": int(checkpoint.replace("chkpt_","").replace(".json","")),
                } | mean_data
            )

    df = pd.DataFrame(df)
    df = df.sort_values(by=["name", "checkpoint"])
    df = df.drop(64, axis=1)
    df = df.drop(32, axis=1)
    return df

def transform_name(x, horizon=25000):
    parts = x.split("_")
    prefix = "_".join(parts[:-1])
    num_str = parts[-1]

    if num_str.isdigit():
        # Compute float value
        value = float("0." + num_str) * horizon
        return f"{prefix}, {value}", value
    else:
        # Keep original, no transformation
        return x, None

CACHE_FILE = "model_size_cache.json"
def _load_model_size_cache(path: str = CACHE_FILE) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            raw = json.load(f)
        # JSON makes keys strings; convert nested keys back to int
        return {m: {int(k): {int(a): b for a, b in v.items()} for k, v in sizes.items()} for m, sizes in raw.items()}
    return {}

def _save_model_size_cache(cache: dict, path: str = CACHE_FILE):
    with open(path, "w") as f:
        json.dump(cache, f)

model_size_cache = _load_model_size_cache()

def get_flops(all_counts, train_rec, train_grad_steps, this_chkpt, max_steps, batch_size=2**20, warmup_duration=0.25, n_only=True, k_only=False, n_and_k=False): # 1M batch size
    if train_rec == 1:
        return all_counts[train_rec][1]["flops_times_by_6d"] * 6 * this_chkpt * batch_size
    else:
        if n_only:
            warmups = warmup_hist(this_chkpt, warmup_duration=warmup_duration, max_steps=max_steps, max_rec=train_rec)
            flops_times_checkpoints = sum(all_counts[k][train_grad_steps]["flops_times_by_6d"] * v for k, v in warmups.items())
            return flops_times_checkpoints * 6 * batch_size
        elif k_only:
            warmups = warmup_hist(this_chkpt, warmup_duration=warmup_duration, max_steps=max_steps, max_rec=train_grad_steps)
            flops_times_checkpoints = sum(all_counts[train_rec][k]["flops_times_by_6d"] * v for k, v in warmups.items())
            return flops_times_checkpoints * 6 * batch_size
        elif n_and_k:
            assert False, "n and k not coded"
        else:
            assert False, "one of n_only, k_only, n_and_k must be on"


def get_flops_counts(model_path):
    global model_size_cache
    dict_name = ("/".join(model_path.split("/")[:-1]))
    if dict_name not in model_size_cache.keys():
        print(f"Adding {dict_name} to model size cache")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        if ("TinyLlama-1.1B-intermediate-step-1431k-3T" in model_path) or ("non_recur" in model_path) or ("Llama-3.2-1B-untied" in model_path):
            # total_params = sum(p.numel() for p in model.parameters())
            total_params = sum(
                p.numel() for name, p in model.named_parameters()
                if not (name.startswith("transformer.wte") or name.startswith("lm_head"))
            )
            model_size_cache[dict_name] = {1: {1: {"total_not_emb_or_lm_head": total_params, "flops_times_by_6d": total_params}}}
        else:
            counts = count_params(model)
            size_dict = {}
            for n in range(1,33):
                this_dict = {}
                for k in range(1,9):
                    this_dict[k] = count_params_with_rec(copy.deepcopy(counts), num_rec=n, num_grad_rec=k)
                size_dict[n] = this_dict
            # size_dict = {k: {8: count_params_with_rec(copy.deepcopy(counts), num_rec=k, num_grad_rec=8)} for k in range(1,33)}
            model_size_cache[dict_name] = size_dict

def schedule_rec_ablation(df, plot_all=True):
    global model_size_cache

    import_times_new_roman(font_size=24)

    multi_rec_df = multi_recurrence_eval_df(df["run_name"].unique())

    df = df[["run_id", "run_name", "step", "avg_sec_per_step"]]
    print(df[df["run_name"] == "1_node_bf16_mixed_mbs_8_bs_16_huginn_llama_1b_2_4_2_last_4_layers_5e_5_n_schedule_only_max_32_warmup_125"]["run_id"].unique())
    # exit()
    df = df.rename(columns={'run_name': 'name', 'step': 'checkpoint'})

    common_df = multi_rec_df.merge(df, on=['name', 'checkpoint'], how='inner')
    print(common_df[common_df["name"] == "1_node_bf16_mixed_mbs_8_bs_16_huginn_llama_1b_2_4_2_last_4_layers_5e_5_n_schedule_only_max_32_warmup_125"]["run_id"].unique())
    # exit()
    common_df['new_name'] = (
        common_df['name']
        .str.replace(
            "1_node_bf16_mixed_mbs_8_bs_16_huginn_llama_1b_2_4_2_last_4_layers_5e_5_", 
            "", 
            regex=False
        )
        .str.replace(r"schedule_(only_)?max_.*?_warmup_", "", regex=True)
        .str.replace("_part_2", "", regex=False)
    )
    common_df = common_df[~common_df['new_name'].str.contains("start", case=False, na=False)]

    # horizon is 25k steps, warmup is fraction of that
    common_df[["new_name", "warmup_steps"]] = common_df["new_name"].apply(
        lambda x: pd.Series(transform_name(x))
    )

    if not plot_all:
        common_df = common_df.drop(columns=[1,8], errors='ignore')
        common_df = common_df[common_df["warmup_steps"].isin([0.0, 6250.0])]

    print(common_df.head())
    # count_params, count_params_with_rec, warmup_hist
    for run, checkpoint in common_df[["name", "checkpoint"]].drop_duplicates().itertuples(index=False):
        try:
            get_flops_counts(f"huginn_llama/{run}/model_only_chkpt_{checkpoint}")
        except:
            print(f"failed for {run}")
    _save_model_size_cache(model_size_cache)

    max_steps = 25000
    def compute_flops(row):
        return get_flops(
            model_size_cache[f"huginn_llama/{row['name'].replace('_part_2','')}"],
            32,
            8,
            row['checkpoint'],
            max_steps,
            batch_size=2**19,
            warmup_duration=row["warmup_steps"] / max_steps,
            n_only="n, " in row["new_name"],
            k_only="k, " in row["new_name"],
        )
    common_df = common_df[
        (common_df["new_name"].str.startswith("n, ")) |
        (common_df["new_name"].str.startswith("k, "))
    ] # no n and k
    common_df["FLOPs"] = common_df.apply(compute_flops, axis=1)
    common_df = common_df.drop(columns=["name", "warmup_steps"]).rename(columns={"new_name": "name"})
    print(common_df.head())

    df_melted = common_df.melt(id_vars=['run_id', 'name', 'checkpoint', "avg_sec_per_step", "FLOPs"], var_name='linesize', value_name='value')
    df_melted['linesize'] = df_melted['linesize'].astype(str)
    df_melted["time_hrs"] = (df_melted['avg_sec_per_step'] * df_melted['checkpoint']) / (60 * 60)
    # print(df_melted.head())
    # exit()
    linestyles = {
        '1': '-.', '2': '-', '4': '--', '8': (0, (3, 1, 1, 1)),
        '16': ':'#, '32': (0, (5, 1)), '64': (0, (1, 1))
    }

    colors = cycle(sns.color_palette("colorblind"))#cycle(plt.cm.tab10.colors)
    model_colors = {model: next(colors) for model in df_melted['name'].unique()}
    all_df = df_melted.copy()
    for n_or_k in ["n", "k"]:
        df_melted = all_df[all_df["name"].str.startswith(f"{n_or_k}, ")]
        # Store used models and linesizes for legends
        used_models = set()
        used_linesizes = set()

        fig, axes = plt.subplots(1, 2, figsize=(20, 3.5), sharey=True, gridspec_kw={'wspace': 0.08})
        ax1, ax2 = axes
        lw = 3

        for (model, linesize), group in df_melted.groupby(['name', 'linesize']):
            ax1.plot(
                group['checkpoint'],
                group['value'],
                label=f'model={model}, linesize={linesize}',
                linestyle=linestyles.get(linesize, '-'),
                color=model_colors[model],
                linewidth=lw
            )
            ax2.plot(
                group["FLOPs"],#['time_hrs'],
                group['value'],
                label=f"model={model}, linesize={linesize}",
                linestyle=linestyles.get(linesize, '-'),
                color=model_colors[model],
                linewidth=lw
            )
            used_models.add(model)
            used_linesizes.add(linesize)
        

        # Create custom legends
        legend_lw = 4
        model_legend_handles = [
            mlines.Line2D([], [], color=model_colors[model], label=round(float(model.replace('n, ','').replace('k, ',''))), linestyle='-', linewidth=legend_lw)
            for model in sorted(used_models)
        ]
        used_linesizes = {ls for ls in used_linesizes if str(ls).isdigit()}
        linestyle_legend_handles = [
            mlines.Line2D([], [], color='black', label=f'{ls}', linestyle=linestyles[ls], linewidth=legend_lw)
            for ls in sorted(used_linesizes, key=int)
        ]
        
        # Plot details
        ax1.set_xlabel("Train Step")
        # ax2.set_xlabel("Time (Hours)")
        ax2.set_xlabel("FLOPs")
        ax1.set_ylabel("Loss")

        ax1.yaxis.set_major_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))
        ax1.yaxis.set_minor_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))

        for ax in axes.flatten():
            ax.set_yscale('log')
            ax.grid(True, axis='both', which='both')
            if not plot_all:
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
                ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter('%.1f'))

        # Add custom legends below the plot
        fig.legend(handles=model_legend_handles, title="Number of Steps in Curriculum" if n_or_k == "n" else "Maximum Backprop Depth", loc='lower center', bbox_to_anchor=(0.3, 0.85), ncol=3, frameon=False)
        fig.legend(handles=linestyle_legend_handles, title="Test Recurrence", loc='lower center', bbox_to_anchor=(0.72, 0.85), ncol=3 if plot_all else 5, frameon=False)
        plt.savefig(f"schedule_rec_ablation_{n_or_k}{'_all' if plot_all else ''}.pdf", bbox_inches="tight")
        plt.close()

def generic_eval_plotter(df, save_name, single_legend=False, x_axes=None, n_cols_in_model_legend=1, split_legend=False):
    # num_rec_styles = {
    #     1: '-',
    #     2: '--',
    #     4: '-.',
    #     8: ':',
    #     16: (0, (3, 1, 1, 1)),  # dash-dot-dot
    #     32: (0, (1, 1)),        # densely dotted
    #     # 64: '-',        # densely dotted
    # }
    num_rec_markers = {
        1: "o",   # circle
        2: "s",   # square
        4: "D",   # diamond
        8:  "X",  # x-filled
        16: "^",  # triangle up
        32: "P",  # plus-filled
        # 64: "v",   # triangle down
    }

    for task_name in df["task"].unique():
        if x_axes is None:
            x_axes = ["chkpt"]
            if task_name == "gsm8k":
                x_axes = ["chkpt", "FLOPs", "effective_params"]
        for x_axis in x_axes:
            subset = df[df['task'] == task_name]
            print(f"{task_name}: {subset['model'].unique()}")

            plt.figure(figsize=(8, 4))
            plt.grid()

            # Assign consistent color per model
            models = sorted(subset["model"].unique())
            cmap = sns.color_palette("colorblind")
            model_colors = {m: cmap[i % len(cmap)] for i, m in enumerate(models)}

            for model_name in models:
                model_subset = subset[subset["model"] == model_name]
                for num_rec in sorted(model_subset["num_rec"].unique()):
                    # style = num_rec_styles.get(num_rec, '-')  # fallback to solid
                    marker = num_rec_markers.get(num_rec, "o")
                    data = model_subset[model_subset["num_rec"] == num_rec].sort_values(x_axis)

                    plt.errorbar(
                        data[x_axis],
                        data["acc"],
                        # yerr=data["stderr"],
                        label=f"{model_name}",# (test rec={num_rec})",
                        color=model_colors[model_name],
                        marker=marker,
                        markersize=8,
                        linestyle='-', 
                        capsize=4,
                    )

            # plt.axhline(y=task_to_random[task_name], color='black', linestyle='--', label='Random Baseline')

            # plt.title(f"Accuracy vs Checkpoint for Task: {task_name}")
            x_labels_dict = {
                "effective_params": "Effective Parameters",
                "FLOPs": "FLOPs",
                "chkpt": "Step",
            }
            plt.xlabel(x_labels_dict[x_axis] if x_axis in x_labels_dict.keys() else x_axis)
            plt.ylabel("Accuracy")

            if single_legend:
                second_legend = plt.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.2),
                    ncol=n_cols_in_model_legend,
                    frameon=False
                )
            else:
                # Legend for models (color)
                model_legend_handles = [
                    matplotlib.lines.Line2D([0], [0], color=model_colors[model], lw=4, label=model)
                    for model in models
                ]
                # Legend for num_rec (linestyle)
                num_rec_legend_handles = [
                    # matplotlib.lines.Line2D([0], [0], color='black', lw=2, linestyle=style, label=f"num_rec={num_rec}")
                    # for num_rec, style in num_rec_styles.items()
                    matplotlib.lines.Line2D([0], [0], color='black', marker=marker, linestyle='None', label=f"{num_rec}", markersize=8)
                    for num_rec, marker in num_rec_markers.items()
                ]

                if split_legend:
                    second_legend = plt.legend(
                        handles=num_rec_legend_handles,
                        title="Test Rec (Marker)",
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        ncol=1,
                        frameon=False,
                        columnspacing=0.6,
                        handletextpad=0.3
                    )
                    plt.gca().add_artist(second_legend)

                    first_legend = plt.legend(
                        handles=model_legend_handles,
                        title="Model (Color)",
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.2),
                        ncol=n_cols_in_model_legend,
                        frameon=False
                    )
                    # plt.gca().add_artist(first_legend)
                else:
                    second_legend = plt.legend(
                        handles=num_rec_legend_handles,
                        title="Test Rec (Marker)",
                        loc='upper center',
                        bbox_to_anchor=(0.75, -0.15),
                        ncol=3,
                        frameon=False,
                        columnspacing=0.6,
                        handletextpad=0.3
                    )
                    plt.gca().add_artist(second_legend)

                    first_legend = plt.legend(
                        handles=model_legend_handles,
                        title="Model (Color)",
                        loc='upper center',
                        bbox_to_anchor=(0.2, -0.15),
                        ncol=n_cols_in_model_legend,
                        frameon=False
                    )
                    # plt.gca().add_artist(first_legend)

            axis_name = f"_{x_axis}" if x_axis != "chkpt" else ""
            plt.savefig(f"{save_name}_{task_name}{axis_name}.pdf", bbox_inches="tight", bbox_extra_artists=(second_legend,) if single_legend else (first_legend, second_legend), transparent=True)
            plt.clf()

def two_panel_eval_plotter_by_axis_df(
    dataframe_by_axis,                     # {"FLOPs": df_flops, "effective_params": df_eff}
    save_name,
    axes=("FLOPs", "effective_params"),
    n_columns_in_model_legend=2,
    figure_size=(12, 4),
):
    num_rec_markers = {1:"o", 2:"s", 4:"D", 8:"X", 16:"^", 32:"P"}
    axis_labels = {"effective_params":"Effective Parameters", "FLOPs":"FLOPs", "chkpt":"Step"}

    tasks = set().union(*[set(dataframe_by_axis[a]["task"].unique()) for a in axes if a in dataframe_by_axis])
    for task_name in sorted(tasks):
        subsets = {a: dataframe_by_axis[a][dataframe_by_axis[a]["task"] == task_name] for a in axes}
        models = sorted(set().union(*[set(s["model"].unique()) for s in subsets.values() if not s.empty]))
        if not models: 
            continue
        num_recs = sorted(set().union(*[set(s["num_rec"].unique()) for s in subsets.values() if not s.empty]))
        cmap = sns.color_palette("colorblind")
        model_colors = {m: cmap[i % len(cmap)] for i, m in enumerate(models)}

        fig, axes_objects = plt.subplots(1, 2, figsize=figure_size, sharey=True)

        for ax in axes_objects:
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

        for axis_object, axis_name in zip(axes_objects, axes):
            subset = subsets[axis_name]
            axis_object.grid(True)
            axis_object.set_xlabel(axis_labels.get(axis_name, axis_name))
            if subset.empty:
                axis_object.text(0.5, 0.5, f"No {axis_name} data", ha="center", va="center", transform=axis_object.transAxes)
                continue
            for model in models:
                model_subset = subset[subset["model"] == model]
                for num_rec in sorted(model_subset["num_rec"].unique()):
                    data = model_subset[model_subset["num_rec"] == num_rec].sort_values(axis_name)
                    axis_object.errorbar(
                        data[axis_name], data["acc"],
                        color=model_colors[model],
                        marker=num_rec_markers.get(num_rec, "o"),
                        markersize=10 if axis_name == "effective_params" else 8, linestyle="-", capsize=4,
                    )
        axes_objects[0].set_ylabel("Accuracy")

        # model_handles = [matplotlib.lines.Line2D([0],[0], color=model_colors[m], lw=4, label=m) for m in models]
        # fig.legend(handles=model_handles, title="Model (Color)", loc="upper center", bbox_to_anchor=(0.3, 0.1), ncol=n_columns_in_model_legend, frameon=False)
        labels = models
        label_colors = [model_colors[m] for m in labels]

        # create invisible dummy handles so only text shows
        dummy_handles = [matplotlib.lines.Line2D([], [], linestyle="None") for _ in labels]

        leg = fig.legend(
            dummy_handles, [l.replace("\n"," ") for l in labels],
            title="Model (Text Color)",
            loc="upper center",
            bbox_to_anchor=(0.3, 0.1),
            ncol=1,
            frameon=False,
            handlelength=0,      # hide the (nonexistent) handle
            handletextpad=0.0,   # tighten spacing
        )

        # color each label individually
        for txt, c in zip(leg.get_texts(), label_colors):
            txt.set_color(c)
            txt.set_fontweight("bold")
            txt.set_fontproperties(font_manager.FontProperties(weight="bold"))
        leg.get_title().set_fontweight("bold")

        # model_handles = [
        #     matplotlib.patches.Patch(color="none", label=m.replace("\n"," ")) for m in models
        # ]

        # legend = fig.legend(
        #     handles=model_handles,
        #     title="Model (Color)",
        #     loc="upper center",
        #     bbox_to_anchor=(0.3, 0.1),
        #     # bbox_to_anchor=(0.5, 0.1),
        #     ncol=1,
        #     frameon=False,
        #     # columnspacing=0.2
        # )

        # # Now recolor the background of the text itself
        # for text, m in zip(legend.get_texts(), models):
        #     text.set_backgroundcolor(model_colors[m])
        #     text.set_color("white")  # make text visible against dark colors

        # legend 2 — num recs
        rec_handles = [matplotlib.lines.Line2D([0],[0], color="black", marker=num_rec_markers.get(nr, "o"), linestyle="None", label=str(nr), markersize=10)for nr in num_recs]
        fig.legend(handles=rec_handles, title="Test Recurrence (Marker)", loc="upper center", bbox_to_anchor=(0.75, 0.1), ncol=3, frameon=False)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.24)
        fig.savefig(f"{save_name}_{task_name}_{axes[0]}_{axes[1]}.pdf", bbox_inches="tight")
        plt.close(fig)

def two_panel_eval_plotter_by_axis_df_2(
    dataframe_by_axis,                     # {"FLOPs": df_flops, "effective_params": df_eff}
    save_name,
    axes=("FLOPs", "effective_params"),
    n_columns_in_model_legend=2,
    figure_size=(12, 4),
):
    if "FLOPs (1e21)" in axes:
        df = dataframe_by_axis.pop("FLOPs")
        df["FLOPs (1e21)"] = df["FLOPs"] / 1e21
        dataframe_by_axis["FLOPs (1e21)"] = df

    num_rec_markers = {1:"o", 2:"s", 4:"D", 8:"X", 16:"^", 32:"P"}
    axis_labels = {"effective_params":"Effective Parameters", "FLOPs":"FLOPs", "chkpt":"Train Step"}

    tasks = set().union(*[set(dataframe_by_axis[a]["task"].unique()) for a in axes if a in dataframe_by_axis])
    for task_name in sorted(tasks):
        subsets = {a: dataframe_by_axis[a][dataframe_by_axis[a]["task"] == task_name] for a in axes}
        models = sorted(set().union(*[set(s["model"].unique()) for s in subsets.values() if not s.empty]))
        if not models: 
            continue
        num_recs = sorted(set().union(*[set(s["num_rec"].unique()) for s in subsets.values() if not s.empty]))
        cmap = sns.color_palette("colorblind")
        model_colors = {m: cmap[i % len(cmap)] for i, m in enumerate(models)}

        fig, axes_objects = plt.subplots(1, 2, figsize=figure_size, sharey=True)

        for ax in axes_objects:
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.grid(True)

        # ---------------------------
        # Left panel: first axis in `axes`
        # ---------------------------
        axis_object_left, axis_name_left = axes_objects[0], axes[0]
        subset_left = subsets[axis_name_left]
        axis_object_left.set_xlabel(axis_labels.get(axis_name_left, axis_name_left))
        axis_object_left.set_ylabel("Accuracy")

        if subset_left.empty:
            axis_object_left.text(0.5, 0.5, f"No {axis_name_left} data", ha="center", va="center", transform=axis_object_left.transAxes)
        else:
            # Keep original behavior for non-effective_params left panel (e.g., FLOPs)
            for model in models:
                model_subset = subset_left[subset_left["model"] == model]
                if model_subset.empty:
                    continue
                for num_rec in sorted(model_subset["num_rec"].unique()):
                    data = model_subset[model_subset["num_rec"] == num_rec].sort_values(axis_name_left)
                    axis_object_left.errorbar(
                        data[axis_name_left], data["acc"],
                        color=model_colors[model],
                        marker=num_rec_markers.get(num_rec, "o"),
                        markersize=8, linestyle="-", capsize=4,
                        label=None
                    )

        # ---------------------------
        # Right panel: effective_params → x-axis becomes num_rec
        # ---------------------------
        axis_object_right = axes_objects[1]
        subset_eff = subsets.get("effective_params", pd.DataFrame())
        if subset_eff.empty:
            axis_object_right.set_xlabel("Test Recurrence")
            axis_object_right.text(0.5, 0.5, "No effective_params data",
                                   ha="center", va="center",
                                   transform=axis_object_right.transAxes)
        else:
            axis_object_right.set_xlabel("Test Recurrence")
            axis_object_right.set_xscale("log")   # <-- log scale here

            # Ensure recurrence ticks are shown nicely
            if len(num_recs) > 0:
                axis_object_right.set_xticks(num_recs)
                axis_object_right.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
                axis_object_right.get_xaxis().set_minor_formatter(mticker.NullFormatter())

            # Plot each model across recurrence on x-axis
            x_min = min(num_recs) if len(num_recs) else 1
            x_max = max(num_recs) if len(num_recs) else 1

            for model in models:
                msub = subset_eff[subset_eff["model"] == model]
                if msub.empty:
                    continue

                is_static_depth = ("static depth" in str(model).lower()) or ("non-recurrent" in str(model).lower())

                by_nr = (msub.groupby("num_rec", as_index=False)
                              .agg(acc=("acc", "mean"))
                         ).sort_values("num_rec")

                if is_static_depth:
                    y = float(by_nr["acc"].mean())
                    axis_object_right.hlines(
                        y=y, xmin=x_min, xmax=x_max,
                        colors=[model_colors[model]],
                        # linestyles=(0, (10, 8)) if model.replace("\n"," ") == "TinyLlama Non-Recurrent - Two Phase" else "-",
                        linestyles="-",
                        linewidth=3
                    )
                    axis_object_right.plot([], [], color=model_colors[model], label=model)
                else:
                    axis_object_right.errorbar(
                        by_nr["num_rec"], by_nr["acc"],
                        color=model_colors[model],
                        marker="o", linestyle="-", capsize=4, markersize=10, lw=3
                    )

        # ---------------------------
        # Legend: models only (text colored). No recurrence legend anymore.
        # ---------------------------
        model_handles = [matplotlib.lines.Line2D([0],[0], color=model_colors[m], lw=4, label=m.replace("\n"," ")) for m in natsorted(models)]
        fig.legend(handles=model_handles, title="Model (Color)", loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=n_columns_in_model_legend, frameon=False)
        
        # labels = models
        # label_colors = [model_colors[m] for m in labels]
        # dummy_handles = [matplotlib.lines.Line2D([], [], linestyle="None") for _ in labels]

        # leg = fig.legend(
        #     dummy_handles, [l.replace("\n"," ") for l in labels],
        #     title="Model (Color)",
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 0.08),
        #     ncol=n_columns_in_model_legend,
        #     frameon=False,
        #     handlelength=0,
        #     handletextpad=0.0,
        # )
        # for txt, c in zip(leg.get_texts(), label_colors):
        #     txt.set_color(c)
        #     txt.set_fontweight("bold")
        #     txt.set_fontproperties(font_manager.FontProperties(weight="bold"))
        # leg.get_title().set_fontweight("bold")

        # Layout & save
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.20)
        fig.savefig(f"{save_name}_{task_name}_{axes[0]}_{axes[1]}.pdf", bbox_inches="tight", transparent=True)
        plt.close(fig)

def gsm8k_plotter(df, save_prefix="", layers_to_pick=[4,8,16,32], last_chkpt=48_000):
    import_times_new_roman(font_size=22)
    print(df)
    # generic_eval_plotter(df, f"{save_prefix}test", x_axes=["FLOPs", "effective_params"])

    filtered_df = df[
        ((df["train_rec"] == df["num_rec"])) &
        (df["train_rec"].isin([1, 4, 16]))
    ]
    generic_eval_plotter(filtered_df, f"{save_prefix}", single_legend=True, x_axes=["FLOPs"])
    filtered_df = df[
        (df["chkpt"] == last_chkpt) 
    ]
    generic_eval_plotter(filtered_df, f"{save_prefix}", single_legend=False, x_axes=["effective_params"], n_cols_in_model_legend=2, split_legend=True)

    dfs = {
        "FLOPs": df[((df["train_rec"] == df["num_rec"])) & (df["train_rec"].isin([1] + layers_to_pick))],
        "effective_params": df[(df["chkpt"] == last_chkpt) & (df["train_rec"].isin([1] + layers_to_pick))],
    }
    # two_panel_eval_plotter_by_axis_df(dfs, f"{save_prefix}combined_plot", figure_size=(14,4))
    two_panel_eval_plotter_by_axis_df_2(dfs, f"{save_prefix}combined_plot", figure_size=(14,3.5), n_columns_in_model_legend=3, axes=("FLOPs (1e21)", "effective_params"))
    dfs = {
        "FLOPs": df[df["train_rec"] == df["num_rec"]],
        "effective_params": df[(df["chkpt"] == last_chkpt)],
    }
    two_panel_eval_plotter_by_axis_df_2(dfs, f"{save_prefix}combined_plot_all", figure_size=(14,4), n_columns_in_model_legend=3, axes=("FLOPs (1e21)", "effective_params"))
    
    import_times_new_roman(font_size=20)
    unique_names = df[df["train_rec"] > 1][["model", "train_rec"]].drop_duplicates()
    print(unique_names)
    for _, (name, rec) in unique_names.iterrows():
        filtered_df = df[
            (df["model"] == name)|
            (df["train_rec"] == 1)
        ]
        generic_eval_plotter(filtered_df, f"{save_prefix}{rec}", single_legend=False, x_axes=["FLOPs"])
    plt.close()
    exit()

def plot_all_evals(df, save_name):
    import_times_new_roman(font_size=28)

    tasks = ['arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'mmlu', 'piqa', 'openbookqa', 'social_iqa'] #df["task"].unique()
    tasks_to_title = {'arc_easy': "Arc-E", 'arc_challenge': "Arc-C", 'hellaswag': "Hellaswag", 'winogrande': "Winogrande", 'mmlu': "MMLU", 'piqa': "PIQA", 'openbookqa': "OBQA", 'social_iqa': "SIQA"}
    n_tasks = len(tasks)
    fig, axes = plt.subplots(2, n_tasks//2, figsize=(20, 8))

    axes = axes.flatten()

    models = df["model"].unique()
    cmap = sns.color_palette("colorblind", len(models))
    color_map = {model: cmap[i] for i, model in enumerate(models)}

    for ax, task in zip(axes, tasks):
        subdf = df[df["task"] == task]
        for model, mdf in subdf.groupby("model"):
            mdf = mdf.sort_values("chkpt")
            ax.plot(
                mdf["chkpt"], mdf["acc"],
                marker="P",
                markersize=8,
                linewidth=3,
                label=f"{model}",
                color=color_map[model]
            )
        ax.set_title(tasks_to_title[task])
        ax.grid()


    legend_proxies = []
    legend_labels = []

    for model in models:
        legend_proxies.append(
            matplotlib.lines.Line2D([0], [0],
                linestyle='-',
                marker='P',
                color=color_map[model],
                linewidth=4,
                markersize=16)
        )
        legend_labels.append(f"{model}")

    fig.legend(
        legend_proxies, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(models),
        frameon=False
    )
    for ax in axes[-(n_tasks // 2):]:
        ax.set_xlabel("Train Step")
    
    for ax in axes[::n_tasks//2]:  
        ax.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(f"{save_name}.pdf", bbox_inches="tight")

def data_mix(df, save_prefix):
    import_times_new_roman(font_size=22)
    dfs = {
        "chkpt": df[df["train_rec"] == df["num_rec"]],
        "effective_params": df[(df["chkpt"] == 25_000)],
    }
    two_panel_eval_plotter_by_axis_df_2(dfs, f"{save_prefix}combined_plot", figure_size=(14,3.5), axes=("chkpt", "effective_params"))

def generic_eval_plotter_with_times_new_roman_call(*args, **kwargs):
    import_times_new_roman(font_size=22)
    generic_eval_plotter(*args, **kwargs)

def bar_plotter(tinyllama_df, llama_df, olmo_df, recs_to_plot=[32,1]): # recs_to_plot must be decreasing
    import_times_new_roman(font_size=20)

    def take_last_chkpt_only(df, chkpt):
        df = df[df["chkpt"] == chkpt]
        # df = df[df["num_rec"] == df["train_rec"]]
        # df = df[(df["num_rec"] == 1) | (df["num_rec"] == 32)]
        df = df[df["num_rec"].isin(recs_to_plot)]
        return df
    tinyllama_df = take_last_chkpt_only(tinyllama_df, 50_000)
    llama_df = take_last_chkpt_only(llama_df, 48_000)
    olmo_df = take_last_chkpt_only(olmo_df, 48_000)

    tinyllama_df["model_family"] = "TinyLlama"
    llama_df["model_family"] = "Llama"
    olmo_df["model_family"] = "Olmo"

    combined = pd.concat([tinyllama_df, llama_df, olmo_df])

    # Simplify recurrence label
    combined["rec_label"] = combined.apply(
        lambda x: "Non Rec" if "Non-Recurrent" in x["model"] else f"Train Rec {int(x['train_rec'])}", axis=1
    )

    # Plot setup
    fig, ax = plt.subplots(figsize=(15, 3))
    palette = sns.color_palette("colorblind")
    colors = {"gsm8k_cot_sean": palette[0], "minerva_math": palette[2]}
    task_to_legend_name  = {"gsm8k_cot_sean": "GSM8K", "minerva_math": "MATH"}
    bar_width = 0.45
    gap = 0.3  # space between model families
    xpos = 0
    xticks = []
    xtick_labels = []
    section_boundaries = []

    # Iterate through families
    for fam in ["TinyLlama", "Llama", "Olmo"]:
        sub = combined[combined["model_family"] == fam]
        recs = sub["train_rec"].unique()
        recs = sorted(recs)#, key=lambda x: 0 if x=="Non-Recurrent" else int(x.split("=")[-1]))

        start_pos = xpos
        for rec in recs:
            legend_name = sub[sub["train_rec"] == rec].iloc[0]["rec_label"]

            for i, task in enumerate(["gsm8k_cot_sean", "minerva_math"]):
                # df_task = sub[(sub["rec_label"] == rec) & (sub["task"] == task)]
                df_task = sub[(sub["train_rec"] == rec) & (sub["task"] == task)]
                for test_rec in recs_to_plot:
                    df_task_test = df_task[df_task["num_rec"] == test_rec]
                    if not df_task_test.empty:
                        ax.bar(
                            xpos + i * bar_width, df_task_test["acc"].values[0],
                            hatch = "//" if test_rec ==32 else "",
                            width=bar_width, color=colors[task],
                            label=task_to_legend_name[task] if fam=="TinyLlama" and rec==1 else "",
                            edgecolor="black", alpha=0.7 if test_rec == 32 else 1.0
                        )
            xticks.append(xpos + bar_width / 2)
            xtick_labels.append(legend_name)
            xpos += 1

        end_pos = xpos - 1
        section_boundaries.append((start_pos, end_pos))
        xpos += gap  # leave gap before next family

    # Vertical dashed lines between sections
    for i in range(1, len(section_boundaries)):
        line_x = section_boundaries[i-1][1] + (bar_width * 1.5) + (gap * 0.66)
        ax.axvline(line_x, color="gray", linestyle="--", alpha=0.6)

    # Add model family titles above each section
    x_min = section_boundaries[0][0] - 0.5
    x_max = section_boundaries[-1][1] + bar_width + 0.5
    ax.set_xlim(x_min, x_max)
    ylim = ax.get_ylim()[1]
    for (start, end), fam in zip(section_boundaries, ["TinyLlama", "Llama", "Olmo"]):
        mid = (start + end) / 2
        ax.text(mid, ylim * 1.02, fam, ha="center", va="bottom", fontsize=26)

    # Formatting
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")

    hatch_patches = [
        matplotlib.patches.Patch(facecolor="white", hatch="//", edgecolor="black", label="32"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", label="1"),
    ]
    hatch_legend = ax.legend(
        handles=hatch_patches,
        title="Test Recurrence",
        loc="upper center",
        bbox_to_anchor=(0.75, -0.55),
        ncol=2,
    )
    ax.add_artist(hatch_legend)

    ax.legend(title="Task", loc="upper center", bbox_to_anchor=(0.33, -0.55), ncol=2)

    # plt.tight_layout()
    ax.yaxis.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    fig.savefig(f"bar_plot.pdf", bbox_inches="tight", transparent=True)#, bbox_extra_artists=(hatch_legend,))


if __name__ == "__main__":
    df = pd.read_csv("plot_1.csv")
    plot_1_extended_lines(df)

    df = pd.read_csv("muon_vs_adam.csv")
    muon(df)

    df = pd.read_csv("schedule_rec_ablation.csv")
    schedule_rec_ablation(df)
    schedule_rec_ablation(df, plot_all=False)
    schedule_rec_explainer(df) # now use scheduling_options.py instead

    df = pd.read_csv("which_layers_llama_1b.csv")
    which_layers(df)

    df = pd.read_csv("shortpgt.csv")
    shortgpt(df)

    def replace_strings(df, non_recurrent_model_name):
        df["model"] = df["model"].str.replace(non_recurrent_model_name, " Non-Recurrent", regex=False)
        df["model"] = df["model"].str.replace(" 75 sqrt", "", regex=False)
        return df
    tinyllama_df = pd.read_json("data/tinyllama_gsm8k_ablate_rec_long.jsonl", lines=True, engine="pyarrow")
    tinyllama_df = replace_strings(tinyllama_df, "-1.1b-3T\nStatic Depth")
    gsm8k_plotter(tinyllama_df, save_prefix="tinyllama_", layers_to_pick=[4,16])
    llama_df = pd.read_json("data/tinyllama_gsm8k_llama_long_4_rec.jsonl", lines=True, engine="pyarrow")
    llama_df = replace_strings(llama_df, "-3.2-1b\nStatic Depth")
    gsm8k_plotter(llama_df, save_prefix="llama_", layers_to_pick=[4,16])
    olmo_df = pd.read_json("data/olmo_50k_steps.jsonl", lines=True, engine="pyarrow")
    olmo_df = replace_strings(olmo_df, "-step1907359\nStatic Depth")
    gsm8k_plotter(olmo_df, save_prefix="olmo_", layers_to_pick=[4,16])
    bar_plotter(tinyllama_df, llama_df, olmo_df)


    df = pd.read_json("data/loop_whole_model.jsonl", lines=True, engine="pyarrow")
    df["model"] = df["model"].str.replace("-1.1b-3T\nStatic Depth", "\nNon-Recurrent", regex=False)
    plot_all_evals(df[(df["num_rec"] == 32) | (df["model"]== "TinyLlama\nNon-Recurrent")], "loop_whole_model")

    df = pd.read_json("data/tinyllama_nemotron_fineweb_2.jsonl", lines=True, engine="pyarrow")
    df["model"] = df["model"].str.replace("-1.1b-3T\nStatic Depth", " Non-Recurrent", regex=False)
    df = df[df["model"] != "tinyllama-hf"]
    data_mix(df, "data_mix_")

    df = pd.read_json("data/extend_prelude_coda.jsonl", lines=True, engine="pyarrow")
    generic_eval_plotter_with_times_new_roman_call(df, f"extend_prelude_coda", single_legend=True, x_axes=["FLOPs"], n_cols_in_model_legend=1, split_legend=False)

    df = pd.read_json("data/tinyllama_1_sqrt_app_fig.jsonl", lines=True, engine="pyarrow")
    df = df[(df["num_rec"] == 4) | (df["model"] == "TinyLlama-1.1b-3T\nStatic Depth")]
    df["model"] = df["model"].str.replace("-1.1b-3T\nStatic Depth", " Non-Recurrent", regex=False)
    generic_eval_plotter_with_times_new_roman_call(df, f"tinyllama_1_sqrt_app_fig", single_legend=True, x_axes=["FLOPs"], n_cols_in_model_legend=1, split_legend=False)