#!/usr/bin/env python3
"""
run_experiment.py — Main Experiment Entry Point

CSE-307 Operating Systems Term Paper
Track 1: Learned Page Replacement (Memory Management)
Title:   "Learning-Augmented Page Replacement Under Workload Shifts"

Steps executed:
    1. Generate the two-phase synthetic trace (seed=42).
    2. Run FIFO, LRU, Optimal (Belady's) for each frame count.
    3. Build training samples from Phase-1 trace using Optimal labels.
    4. Train a Decision Tree classifier (evaluate on held-out data).
    5. Run the Learned policy for each frame count.
    6. Write results/raw_results.csv
    7. Write results/phase_results.csv
    8. Write results/model_metrics.csv
    9. Generate figures in results/figures/
   10. Print concise summary to the terminal.

Usage:
    python run_experiment.py

All results are reproducible from the fixed random seed.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; no GUI required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Ensure src/ is importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.workload       import generate_trace
from src.fifo           import run_fifo
from src.lru            import run_lru
from src.optimal        import run_optimal
from src.features       import build_training_samples
from src.learned_policy import train_model, run_learned_policy, FEATURE_NAMES
from src.bonus_explanation import analyze_explanation_confidence, generate_explanation
from src.evaluation     import collect_raw_results, collect_phase_results

# ── Experiment parameters ────────────────────────────────────────────────────
FRAME_COUNTS  = [3, 4, 5, 6]
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
ALGO_NAMES    = ["FIFO", "LRU", "Optimal", "Learned"]
ALGO_COLORS   = {"FIFO": "#e74c3c", "LRU": "#3498db",
                 "Optimal": "#2ecc71", "Learned": "#f39c12"}
ALGO_MARKERS  = {"FIFO": "o", "LRU": "s", "Optimal": "^", "Learned": "D"}

os.makedirs(FIGURES_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def write_csv(filepath, rows):
    if not rows:
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, filename):
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Generate trace
# ════════════════════════════════════════════════════════════════════════════

def step1_generate_trace():
    print("\n── Step 1: Generating trace ──────────────────────────────────")
    trace, shift_index, meta = generate_trace()
    print(f"  Trace length  : {meta['total_length']}")
    print(f"  Phase 1       : indices 0..{shift_index - 1}  ({meta['phase1_desc']})")
    print(f"  Phase 2       : indices {shift_index}..{meta['total_length'] - 1}"
          f"  ({meta['phase2_desc']})")
    print(f"  Page range    : 0 .. {meta['page_range'] - 1}")
    print(f"  Random seed   : {meta['random_seed']}")
    return trace, shift_index, meta


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Run classical algorithms
# ════════════════════════════════════════════════════════════════════════════

def step2_run_classical(trace, frame_counts):
    print("\n── Step 2: Running classical algorithms ──────────────────────")
    results = {}
    for nf in frame_counts:
        results.setdefault("FIFO",    {})[nf] = run_fifo(trace, nf)
        results.setdefault("LRU",     {})[nf] = run_lru(trace, nf)
        results.setdefault("Optimal", {})[nf] = run_optimal(trace, nf)
        print(f"  frames={nf}  FIFO faults={results['FIFO'][nf]['faults']} "
              f" LRU faults={results['LRU'][nf]['faults']} "
              f" Optimal faults={results['Optimal'][nf]['faults']}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# Steps 3–4 — Build training data and train model
# ════════════════════════════════════════════════════════════════════════════

def step34_train_learned_model(trace, shift_index, frame_counts):
    print("\n── Steps 3–4: Training learned policy ───────────────────────")
    # Training data comes from Phase 1 only (locality-heavy).
    # We use a representative frame count (the median) for sample collection
    # so the model generalises across all tested frame counts.
    training_frame_count = frame_counts[len(frame_counts) // 2]

    # Build next-use table for the FULL trace so Optimal labels are correct.
    opt_result = run_optimal(trace, training_frame_count)

    phase1_trace = trace[:shift_index]
    print(f"  Collecting training samples from Phase 1 "
          f"(len={len(phase1_trace)}, frames={training_frame_count}) ...")
    X, y = build_training_samples(
        phase1_trace, training_frame_count, opt_result["next_use"]
    )
    print(f"  Training samples: {len(X)} total  "
          f"(positive={sum(y)}, negative={len(y) - sum(y)})")

    model, metrics, report = train_model(X, y)

    print(f"  Model: DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)")
    print(f"  Train / Test split: {metrics['n_train']} / {metrics['n_test']}")
    print(f"  Test  accuracy : {metrics['accuracy']:.4f}")
    print(f"  Test  precision: {metrics['precision']:.4f}")
    print(f"  Test  recall   : {metrics['recall']:.4f}")
    print(f"  Test  F1       : {metrics['f1']:.4f}")
    print(f"\n  Classification report:\n{report}")

    return model, metrics


# ════════════════════════════════════════════════════════════════════════════
# Step 5 — Run learned policy
# ════════════════════════════════════════════════════════════════════════════

def step5_run_learned(trace, frame_counts, model, results):
    print("\n── Step 5: Running learned policy ───────────────────────────")
    results.setdefault("Learned", {})
    for nf in frame_counts:
        r = run_learned_policy(trace, nf, model)
        results["Learned"][nf] = r
        print(f"  frames={nf}  Learned faults={r['faults']}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# Steps 6–8 — Write CSVs and figures
# ════════════════════════════════════════════════════════════════════════════

def step6_write_csvs(results, frame_counts, shift_index, model_metrics):
    print("\n── Steps 6–8: Writing CSVs ───────────────────────────────────")

    raw_rows   = collect_raw_results(results, frame_counts)
    phase_rows = collect_phase_results(results, frame_counts, shift_index)

    write_csv(os.path.join(RESULTS_DIR, "raw_results.csv"),   raw_rows)
    write_csv(os.path.join(RESULTS_DIR, "phase_results.csv"), phase_rows)

    model_row = [{"metric": k, "value": v} for k, v in model_metrics.items()]
    write_csv(os.path.join(RESULTS_DIR, "model_metrics.csv"), model_row)

    print(f"  results/raw_results.csv   ({len(raw_rows)} rows)")
    print(f"  results/phase_results.csv ({len(phase_rows)} rows)")
    print(f"  results/model_metrics.csv")

    return raw_rows, phase_rows


# ════════════════════════════════════════════════════════════════════════════
# Step 7 — Generate figures
# ════════════════════════════════════════════════════════════════════════════

def step7_generate_figures(trace, shift_index, raw_rows, phase_rows,
                            frame_counts):
    print("\n── Step 7: Generating figures ────────────────────────────────")

    df_raw   = pd.DataFrame(raw_rows)
    df_phase = pd.DataFrame(phase_rows)

    # ── Figure 1: Workload trace visualisation ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)

    ax = axes[0]
    ax.scatter(range(len(trace)), trace, s=2, color="#2c3e50", alpha=0.5)
    ax.axvline(shift_index, color="red", linewidth=1.5,
               linestyle="--", label=f"Workload shift (index {shift_index})")
    ax.set_xlabel("Access Index", fontsize=11)
    ax.set_ylabel("Page Number", fontsize=11)
    ax.set_title("Synthetic Page-Reference Trace with Workload Shift", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-1, 26)

    ax2 = axes[1]
    window = 50
    trace_arr = np.array(trace)
    unique_counts = [
        len(set(trace_arr[max(0, i - window):i + 1]))
        for i in range(len(trace))
    ]
    ax2.plot(unique_counts, color="#8e44ad", linewidth=1)
    ax2.axvline(shift_index, color="red", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Access Index", fontsize=11)
    ax2.set_ylabel(f"Unique pages\n(last {window} accesses)", fontsize=10)
    ax2.set_title("Working-Set Size Over Time (Sliding Window)", fontsize=12)

    fig.tight_layout(pad=1.5)
    savefig(fig, "workload_trace.png")

    # ── Figure 2: Hit-ratio comparison (all frames) ──────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGO_NAMES:
        subset = df_raw[df_raw["algorithm"] == algo]
        ax.plot(
            subset["num_frames"], subset["hit_ratio"],
            marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
            linewidth=2, markersize=8, label=algo,
        )
    ax.set_xlabel("Number of Page Frames", fontsize=12)
    ax.set_ylabel("Hit Ratio", fontsize=12)
    ax.set_title("Hit Ratio vs. Number of Frames\n(Full Trace)", fontsize=13)
    ax.set_xticks(frame_counts)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    savefig(fig, "hit_ratio_comparison.png")

    # ── Figure 3: Page-fault comparison ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGO_NAMES:
        subset = df_raw[df_raw["algorithm"] == algo]
        ax.plot(
            subset["num_frames"], subset["faults"],
            marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
            linewidth=2, markersize=8, label=algo,
        )
    ax.set_xlabel("Number of Page Frames", fontsize=12)
    ax.set_ylabel("Total Page Faults", fontsize=12)
    ax.set_title("Page Faults vs. Number of Frames\n(Full Trace)", fontsize=13)
    ax.set_xticks(frame_counts)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    savefig(fig, "page_fault_comparison.png")

    # ── Figure 4: Phase comparison (frames=4 as representative) ─────────
    rep_frames = 4
    phase_subset = df_phase[df_phase["num_frames"] == rep_frames]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    phase_labels = {1: "Phase 1\n(Locality-heavy)", 2: "Phase 2\n(Random/bursty)"}
    bar_width = 0.2
    x = np.arange(len(ALGO_NAMES))

    for col_idx, (phase_num, phase_label) in enumerate(phase_labels.items()):
        ax = axes[col_idx]
        ph_data = phase_subset[phase_subset["phase"] == phase_num]
        for i, algo in enumerate(ALGO_NAMES):
            row = ph_data[ph_data["algorithm"] == algo]
            if not row.empty:
                ax.bar(
                    i, row["hit_ratio"].values[0],
                    color=ALGO_COLORS[algo], label=algo,
                    width=0.6, alpha=0.85,
                )
                ax.text(
                    i, row["hit_ratio"].values[0] + 0.01,
                    f"{row['hit_ratio'].values[0]:.2f}",
                    ha="center", va="bottom", fontsize=10,
                )
        ax.set_xticks(range(len(ALGO_NAMES)))
        ax.set_xticklabels(ALGO_NAMES, fontsize=11)
        ax.set_ylabel("Hit Ratio", fontsize=12)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{phase_label}\n(frames={rep_frames})", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=ALGO_COLORS[a]) for a in ALGO_NAMES]
    fig.legend(handles, ALGO_NAMES, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Phase-by-Phase Hit Ratio Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "phase_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# Step 8 — Print summary
# ════════════════════════════════════════════════════════════════════════════

def step8_print_summary(results, frame_counts, shift_index):
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Trace length : 1000  |  Workload shift at index : {shift_index}")
    print(f"  Frame counts : {frame_counts}")
    print()

    # ── Overall results table ────────────────────────────────────────────
    header = f"{'Algorithm':<10} {'Frames':>6} {'Hits':>6} {'Faults':>7} "  \
             f"{'HitRatio':>9} {'FaultRatio':>11}"
    print(header)
    print("-" * len(header))
    for algo in ALGO_NAMES:
        for nf in frame_counts:
            r = results[algo][nf]
            print(f"{algo:<10} {nf:>6} {r['hits']:>6} {r['faults']:>7} "
                  f"{r['hit_ratio']:>9.4f} {r['fault_ratio']:>11.4f}")
        print()

    # ── Phase split at representative frame count ────────────────────────
    rep_nf = 4
    print(f"\n  Phase breakdown (frames={rep_nf}):")
    print(f"  {'Algorithm':<10} {'Phase':>6} {'Hits':>6} {'Faults':>7} "
          f"{'HitRatio':>9}")
    print("  " + "-" * 46)
    from src.evaluation import split_phases
    for algo in ALGO_NAMES:
        r = results[algo][rep_nf]
        p1, p2 = split_phases(r, shift_index)
        for label, stats in [("Phase1", p1), ("Phase2", p2)]:
            print(f"  {algo:<10} {label:>6} {stats['hits']:>6} "
                  f"{stats['faults']:>7} {stats['hit_ratio']:>9.4f}")
        print()

    print("=" * 70)
    print("  Output files:")
    print("    results/raw_results.csv")
    print("    results/phase_results.csv")
    print("    results/model_metrics.csv")
    print("    results/figures/workload_trace.png")
    print("    results/figures/hit_ratio_comparison.png")
    print("    results/figures/page_fault_comparison.png")
    print("    results/figures/phase_comparison.png")
    print("=" * 70)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("CSE-307 — Learning-Augmented Page Replacement Under Workload Shifts")
    print("=" * 70)

    trace, shift_index, meta = step1_generate_trace()
    results                  = step2_run_classical(trace, FRAME_COUNTS)
    model, model_metrics     = step34_train_learned_model(
                                   trace, shift_index, FRAME_COUNTS)
    results                  = step5_run_learned(trace, FRAME_COUNTS,
                                   model, results)
    raw_rows, phase_rows     = step6_write_csvs(
                                   results, FRAME_COUNTS,
                                   shift_index, model_metrics)
    step7_generate_figures(trace, shift_index, raw_rows, phase_rows,
                           FRAME_COUNTS)
    step8_print_summary(results, FRAME_COUNTS, shift_index)

    # ── Optional / Bonus Track (+10 Points): Explanation & Calibration ───
    print("\n── Optional Bonus Track: Decision Explanation & Calibration ──")
    rep_frames = 4
    learned_pa = results["Learned"][rep_frames]["per_access"]
    opt_next_use = results["Optimal"][rep_frames].get("next_use", [])
    calib = analyze_explanation_confidence(learned_pa, trace, opt_next_use, num_frames=rep_frames)
    
    if calib:
        print(f"  Representative Frame Count: {rep_frames}")
        print(f"  Total Evictions Analyzed : {calib['total_evictions']}")
        print(f"  Optimal Eviction Matches : {calib['correct_count']} ({calib['accuracy']*100:.1f}%)")
        print(f"  Confidence when Correct  : {calib['mean_confidence_when_correct']:.4f}")
        print(f"  Confidence when Incorrect: {calib['mean_confidence_when_incorrect']:.4f}")
        print(f"  Confidence Well-Calibrated? {calib['is_calibrated']} (Higher confidence on correct decisions)")
        
        # Save bonus metrics
        bonus_rows = [{"metric": k, "value": v} for k, v in calib.items()]
        write_csv(os.path.join(RESULTS_DIR, "bonus_confidence_metrics.csv"), bonus_rows)
        print("  Saved results/bonus_confidence_metrics.csv")

    print("\nDone.  All results are in the results/ directory.\n")


if __name__ == "__main__":
    main()
