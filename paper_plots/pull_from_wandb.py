import wandb
import pandas as pd
import numpy as np

plots = ["plot_1", "muon_vs_adam", "which_layers_llama_1b", "schedule_rec_ablation", "shortpgt"]
plots = ["schedule_rec_ablation"]
api = wandb.Api()

for plot in plots:
    runs = api.runs("smcleish/huginn_llama", filters={"tags": plot})

    records = []
    for run in runs:
        if run.id == "e4g14kji":
            continue
        print(f"Processing run: {run.name} ({run.id})")
        
        for row in run.scan_history():
            if "train/loss" in row:
                records.append({
                    "run_id": run.id,
                    "run_name": run.name,
                    "step": row["_step"],
                    "loss": row["train/loss"],
                    "mean_recurrence": row["train/mean_recurrence"],
                    "mean_backprop_depth": row["train/mean_backprop_depth"],
                    "_runtime": row.get("_runtime"), # seconds since run start
                    "_timestamp": row.get("_timestamp"), # UNIX time (fallback)
                })

    # Convert to DataFrame
    df = pd.DataFrame(records)
    print(df)
    # Sort by run and step for readability
    df.sort_values(by=["run_id", "step"], inplace=True)
    
    if plot == "schedule_rec_ablation":
        time_col = "_runtime" if "_runtime" in df.columns and df["_runtime"].notna().any() else "_timestamp"
        step_d = df.groupby("run_id")["step"].diff()
        time_d = df.groupby("run_id")[time_col].diff()
        df["sec_per_step"] = (time_d / step_d).astype(float)

        mean_of_rates = (
            df.loc[df["sec_per_step"].notna() & (df["sec_per_step"] > 0)]
            .groupby("run_id")["sec_per_step"].mean()
            .rename("avg_sec_per_step")
        )
        df = df.merge(mean_of_rates, on="run_id", how="left")

    df = df.drop(columns=["_runtime", "_timestamp"])

    # Show preview
    print(df.head())
    df.to_csv(f"{plot}.csv")