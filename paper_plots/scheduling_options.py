import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import matplotlib
import sys
from pathlib import Path
from transformers import get_scheduler
import seaborn as sns
import math
import torch
matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(
    color=sns.color_palette("colorblind")
)

def import_times_new_roman(this_font_manager=font_manager, this_plt=plt, font_size=16):
    this_font_manager.fontManager.addfont(f"Times New Roman.ttf")
    this_plt.rcParams["font.family"] = "Times New Roman"
    this_plt.rcParams["font.size"] = font_size

def run_loop(mean_recurrence_scheduler, max_training_steps):
    to_ret = []
    for i in range(max_training_steps):
        new_mean_rec = math.ceil(mean_recurrence_scheduler.get_last_lr()[0])
        to_ret.append(new_mean_rec)
        mean_recurrence_scheduler.step()

    return to_ret

def get_linear_schedule(max_training_steps, warmup_fraction, max_mean_rec, warmup_type="linear"):
    num_warmup_steps = math.ceil(warmup_fraction * max_training_steps)
    mean_recurrence_scheduler = get_scheduler(
        name="warmup_stable_decay",
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=float(max_mean_rec)),
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_training_steps,
        scheduler_specific_kwargs={"num_decay_steps":0, "min_lr_ratio":0, "warmup_type": warmup_type},
    )
    return run_loop(mean_recurrence_scheduler, max_training_steps)

def schedule_rec_explainer(df, save_name="schedule_rec_options", max_steps=6125, warmup_fraction=0.5, max_mean_rec=32):
    # df = df[df["run_name"] == "1_node_bf16_mixed_mbs_8_bs_16_huginn_llama_1b_2_4_2_last_4_layers_5e_5_n_schedule_only_max_32_warmup_125"]
    # df = df[df["step"] <= 6250]
    import_times_new_roman(font_size=24)

    plt.figure(figsize=(8, 4))
    end_count = int(max_steps * warmup_fraction) + 1

    mean_rec = get_linear_schedule(max_training_steps=max_steps, warmup_fraction=warmup_fraction, max_mean_rec=max_mean_rec)
    plt.plot(np.arange(0, max_steps)[1:], mean_rec[1:], linestyle='-', lw=4, label=f"linear, {sum(mean_rec[1:end_count]):,}")
    mean_rec = get_linear_schedule(max_training_steps=max_steps, warmup_fraction=warmup_fraction, max_mean_rec=max_mean_rec, warmup_type="1-sqrt")
    plt.plot(np.arange(0, max_steps)[1:], mean_rec[1:], linestyle='-', lw=4, label=f"1-sqrt, {sum(mean_rec[1:end_count]):,}")


    plt.xlabel('Train Step')
    plt.ylabel('Mean Recurrence')

    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f"{save_name}.pdf", bbox_inches="tight")
    exit()

if __name__ == "__main__":
    df = pd.read_csv("schedule_rec_ablation.csv")
    schedule_rec_explainer(df)
