"""
evaluation.py — Result Collection and Phase-Split Analysis

Provides helpers to:
    1. Split per-access results into Phase 1 and Phase 2 sub-results.
    2. Compute aggregate statistics for any sub-sequence.
    3. Collect results across multiple algorithms and frame counts
       into a tidy format for CSV export.
"""


def _stats_for_slice(per_access_slice):
    """
    Compute hit/fault statistics for a list of per-access dicts.

    Parameters
    ----------
    per_access_slice : list[dict]
        Subset of the per_access list returned by any algorithm runner.

    Returns
    -------
    dict with total_accesses, hits, faults, hit_ratio, fault_ratio.
    """
    total = len(per_access_slice)
    hits   = sum(1 for r in per_access_slice if r["decision"] == "hit")
    faults = total - hits
    return {
        "total_accesses": total,
        "hits":           hits,
        "faults":         faults,
        "hit_ratio":      hits   / total if total > 0 else 0.0,
        "fault_ratio":    faults / total if total > 0 else 0.0,
    }


def split_phases(result, shift_index):
    """
    Split an algorithm result into Phase 1 and Phase 2 sub-results.

    Parameters
    ----------
    result      : dict  — returned by any run_* function
    shift_index : int   — index where Phase 2 begins

    Returns
    -------
    phase1_stats : dict
    phase2_stats : dict
    """
    pa = result["per_access"]
    phase1_stats = _stats_for_slice(pa[:shift_index])
    phase2_stats = _stats_for_slice(pa[shift_index:])
    return phase1_stats, phase2_stats


def collect_raw_results(algorithms_results, frame_counts):
    """
    Flatten results from multiple algorithms and frame counts into a
    list of rows suitable for a DataFrame / CSV.

    Parameters
    ----------
    algorithms_results : dict[str, dict[int, dict]]
        algorithms_results[algo_name][num_frames] = result_dict
    frame_counts : list[int]

    Returns
    -------
    list[dict]  — one dict per (algorithm, frame_count) combination
    """
    rows = []
    for algo, by_frames in algorithms_results.items():
        for nf in frame_counts:
            r = by_frames[nf]
            rows.append({
                "algorithm":      algo,
                "num_frames":     nf,
                "total_accesses": r["total_accesses"],
                "hits":           r["hits"],
                "faults":         r["faults"],
                "hit_ratio":      round(r["hit_ratio"],   4),
                "fault_ratio":    round(r["fault_ratio"], 4),
            })
    return rows


def collect_phase_results(algorithms_results, frame_counts, shift_index):
    """
    Collect phase-split statistics for all algorithms and frame counts.

    Returns
    -------
    list[dict]  — one dict per (algorithm, frame_count, phase)
    """
    rows = []
    for algo, by_frames in algorithms_results.items():
        for nf in frame_counts:
            r = by_frames[nf]
            p1, p2 = split_phases(r, shift_index)
            for phase_num, stats in [(1, p1), (2, p2)]:
                rows.append({
                    "algorithm":      algo,
                    "num_frames":     nf,
                    "phase":          phase_num,
                    "total_accesses": stats["total_accesses"],
                    "hits":           stats["hits"],
                    "faults":         stats["faults"],
                    "hit_ratio":      round(stats["hit_ratio"],   4),
                    "fault_ratio":    round(stats["fault_ratio"], 4),
                })
    return rows
