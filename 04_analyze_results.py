import pandas as pd
import numpy as np

def analyze_results(csv_path="experiment_results.csv"):
    df = pd.read_csv(csv_path)

    print("========================================")
    print(" IT TICKET TRIAGE - RESULTS ANALYSIS")
    print("========================================\n")

    # Group by query, model, batch, CPU, use_gbnf to calculate STABILITY first
    grouped = df.groupby(["model", "batch_condition", "is_cpu", "use_gbnf", "query_id", "ambiguous"])

    # 1. calculate variants (how many unique raw_outputs per condition)
    def count_variants(series):
        return series.nunique()

    # 2. calculate stability (is variant count == 1?)
    stability_df = grouped.agg(
        variant_count=("raw_output", count_variants),
        is_stable=("raw_output", lambda x: x.nunique() == 1),
        
        # Per field stability
        scope_stable=("pred_scope", count_variants),
        sev_stable=("pred_severity", count_variants),
        count_stable=("pred_count", count_variants),
        
        # We also need to know if it's consistently correct.
        # It's consistently correct if it is stable AND the exact_match is True for all reps.
        exact_match_all=("exact_match", "all")
    ).reset_index()

    stability_df["consistent_correct"] = stability_df["is_stable"] & stability_df["exact_match_all"]

    # =========================================================
    # TIER 2: ACCURACY (GBNF vs NO GBNF)
    # =========================================================
    print("--- TIER 2: OVERALL ACCURACY (by Model, Ambiguity, GBNF) ---")
    # For overall accuracy, we can look at the raw DataFrame (all reps)
    acc_df = df.groupby(["model", "use_gbnf", "ambiguous"])["exact_match"].mean().reset_index()
    acc_df["exact_match"] = acc_df["exact_match"] * 100
    print(acc_df.rename(columns={"exact_match": "exact_match_acc_%"}).to_string(index=False))
    print("\n")

    # =========================================================
    # TIER 3 & 5: STABILITY VS BATCH SIZE (The core mechanism)
    # =========================================================
    print("--- TIER 3 & 5: STABILITY BY BATCH SIZE & HARDWARE ---")
    # Exclude CPU from the batch scaling for a pure GPU scaling look
    gpu_stability = stability_df[stability_df["is_cpu"] == False]
    batch_scaling = gpu_stability.groupby(["model", "use_gbnf", "batch_condition"])["is_stable"].mean().reset_index()
    batch_scaling["is_stable"] = batch_scaling["is_stable"] * 100
    
    # Pivot for clean viewing
    pivot_scaling = batch_scaling.pivot(index=["model", "use_gbnf"], columns="batch_condition", values="is_stable")
    print("GPU Stability % (Perfect Match Rate) across Batch Sizes:")
    print(pivot_scaling)
    print("\n")

    # Compare GPU batch=1 vs CPU batch=1
    print("--- CPU vs GPU CONTROL (Batch=1) ---")
    b1_df = stability_df[stability_df["batch_condition"] == 1]
    hw_cmp = b1_df.groupby(["model", "use_gbnf", "is_cpu"])["is_stable"].mean().reset_index()
    hw_cmp["is_stable"] = hw_cmp["is_stable"] * 100
    pivot_hw = hw_cmp.pivot(index=["model", "use_gbnf"], columns="is_cpu", values="is_stable")
    pivot_hw = pivot_hw.rename(columns={False: "GPU_Batch1_%", True: "CPU_Batch1_%"})
    print(pivot_hw)
    print("\n")

    # =========================================================
    # TIER 4: CONSISTENT-CORRECT RATE
    # =========================================================
    print("--- TIER 4: CONSISTENT-CORRECT RATE (by Model, GBNF) ---")
    # This is the headline metric! "Can we trust it in production?"
    # We look at GPU across all batch sizes (since production uses batching)
    trust_df = gpu_stability.groupby(["model", "use_gbnf"])["consistent_correct"].mean().reset_index()
    trust_df["consistent_correct"] = trust_df["consistent_correct"] * 100
    print(trust_df.rename(columns={"consistent_correct": "consistent_correct_% (GPU overall)"}).to_string(index=False))
    print("\n")
    
if __name__ == "__main__":
    analyze_results()

