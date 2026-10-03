"""
bonus_explanation.py — Optional / Bonus Track (+10 Points)
Explanation Confidence Generator and Calibration Analysis

Course Brief:
    "For any of the four tracks above, use a free-tier LLM API (or a simple
     rule-based template generator if no API is available) to produce a short
     natural-language explanation of the algorithm's decision at each step,
     along with a self-rated confidence score. Compare that confidence score
     against whether the decision was actually correct."

How this module works:
    1. During each page eviction, produces a structured natural-language
       explanation justifying the chosen eviction candidate based on its
       recency, historical frequency, and model confidence score.
    2. Compares the chosen candidate against ground-truth (Belady's Optimal).
    3. Evaluates whether the confidence score is well-calibrated (i.e. whether
       the confidence is higher when the prediction is actually correct).
"""

import numpy as np


def generate_explanation(evicted_page, confidence, features_dict, is_correct, optimal_page):
    """
    Generate a human-readable explanation of the eviction decision.
    """
    recency = features_dict.get("recency", 0)
    freq = features_dict.get("frequency", 0)
    
    explanation = (
        f"Evicted Page {evicted_page} (Confidence: {confidence:.2f}): "
        f"The page has been idle for {recency} access steps with total historical frequency {freq}. "
        f"The learned model predicted highest likelihood of long-term staleness."
    )
    if is_correct:
        evaluation = f"CORRECT: Matches Belady's Optimal eviction (Page {optimal_page})."
    else:
        evaluation = f"SUBOPTIMAL: Belady's Optimal would have evicted Page {optimal_page} instead."
        
    return explanation, evaluation


def analyze_explanation_confidence(per_access_log, trace, next_use_table, num_frames=4):
    """
    Extract decisions from the simulation, generate natural-language explanations,
    and measure confidence calibration.
    """
    records = []
    
    for item in per_access_log:
        if item.get("decision") == "fault" and item.get("evicted") is not None:
            evicted = item["evicted"]
            idx = item.get("index", 0)
            frames = item.get("frames_snapshot", set())
            candidates = (set(frames) - {item["page"]}) | {evicted}
            
            # Ground truth: which page would Optimal evict?
            def next_use_of(p):
                for j in range(idx + 1, len(trace)):
                    if trace[j] == p:
                        return j
                return len(trace)
            
            opt_page = max(candidates, key=next_use_of) if candidates else evicted
            is_correct = (evicted == opt_page)
            
            # Confidence score from model prediction
            conf = float(item.get("confidence", 0.50))
            
            records.append({
                "page": evicted,
                "confidence": conf,
                "is_correct": is_correct,
                "opt_page": opt_page,
            })
            
    if not records:
        return {}
        
    correct_confs = [r["confidence"] for r in records if r["is_correct"]]
    incorrect_confs = [r["confidence"] for r in records if not r["is_correct"]]
    
    mean_correct = float(np.mean(correct_confs)) if correct_confs else 0.0
    mean_incorrect = float(np.mean(incorrect_confs)) if incorrect_confs else 0.0
    
    return {
        "total_evictions": len(records),
        "correct_count": len(correct_confs),
        "accuracy": len(correct_confs) / len(records),
        "mean_confidence_when_correct": mean_correct,
        "mean_confidence_when_incorrect": mean_incorrect,
        "is_calibrated": mean_correct > mean_incorrect,
    }
