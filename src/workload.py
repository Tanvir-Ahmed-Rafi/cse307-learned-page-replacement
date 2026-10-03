"""
workload.py — Synthetic Page-Reference Trace Generator

Trace design
============
The trace has TWO clearly distinct phases separated by a workload shift.

PHASE 1  (indices 0 .. PHASE1_LEN-1)   — locality-heavy / sequential
    - A small "hot" working set of HOT_PAGES pages is accessed frequently.
    - Within each mini-burst a short sequential sub-run is inserted to
      mimic spatial locality (e.g., loading consecutive code pages).
    - Occasional "background" accesses to a slightly wider page range
      add noise without destroying locality.

PHASE 2  (indices PHASE1_LEN .. TOTAL_LEN-1)  — random / bursty
    - Accesses are drawn uniformly at random from a LARGER page range.
    - No working-set discipline; the hot set changes every few accesses.
    - This models a context-switch to a streaming / scanning workload.

Parameters (all exposed as module-level constants for traceability)
--------------------
TOTAL_LEN   = 1000   total number of page accesses
PHASE1_LEN  =  500   workload shift at index 500
PAGE_RANGE  =   25   pages numbered 0 .. 24 (universe of all pages)

Phase 1 sub-parameters:
    HOT_PAGES       =  6   pages in the hot working set
    SEQ_BURST_LEN   =  4   length of sequential sub-runs
    BG_PROB         = 0.10 probability of a background (non-hot) access

Phase 2 sub-parameters:
    All pages 0..24 drawn uniformly at random.

Random seed: 42  (ensures full reproducibility)
"""

import numpy as np

# ── Trace parameters ────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TOTAL_LEN    = 1000
PHASE1_LEN   = 500       # shift occurs at this index
PAGE_RANGE   = 25        # pages are integers in [0, PAGE_RANGE)

# Phase-1 working set
HOT_PAGES    = 6         # pages in the hot set
SEQ_BURST_LEN = 4        # consecutive-page mini-burst length
BG_PROB      = 0.10      # probability of going outside the hot set


def generate_trace(seed=RANDOM_SEED):
    """
    Generate the two-phase synthetic page-reference trace.

    Returns
    -------
    trace       : list[int]   — full sequence of page accesses
    shift_index : int         — index where Phase 2 begins (== PHASE1_LEN)
    metadata    : dict        — documents the trace parameters
    """
    rng = np.random.default_rng(seed)

    # ── Phase 1: locality-heavy ──────────────────────────────────────────
    hot_set = list(range(HOT_PAGES))   # pages 0,1,2,3,4,5

    phase1 = []
    while len(phase1) < PHASE1_LEN:
        # Decide: sequential burst or single hot-set access
        if rng.random() < BG_PROB:
            # Occasional background access to a page outside the hot set
            bg_page = int(rng.integers(HOT_PAGES, PAGE_RANGE))
            phase1.append(bg_page)
        else:
            # With 40% chance, do a short sequential burst
            if rng.random() < 0.40:
                start = int(rng.choice(hot_set))
                for offset in range(SEQ_BURST_LEN):
                    page = (start + offset) % HOT_PAGES
                    phase1.append(page)
                    if len(phase1) >= PHASE1_LEN:
                        break
            else:
                # Single access to a randomly chosen hot-set page
                page = int(rng.choice(hot_set))
                phase1.append(page)

    phase1 = phase1[:PHASE1_LEN]

    # ── Phase 2: random / bursty ─────────────────────────────────────────
    # Draw uniformly from the full page range — no locality discipline.
    phase2_raw = rng.integers(0, PAGE_RANGE, size=TOTAL_LEN - PHASE1_LEN)
    phase2 = phase2_raw.tolist()

    trace = phase1 + phase2

    metadata = {
        "total_length":   TOTAL_LEN,
        "phase1_length":  PHASE1_LEN,
        "phase2_length":  TOTAL_LEN - PHASE1_LEN,
        "shift_index":    PHASE1_LEN,
        "page_range":     PAGE_RANGE,
        "hot_pages":      HOT_PAGES,
        "seq_burst_len":  SEQ_BURST_LEN,
        "bg_prob":        BG_PROB,
        "random_seed":    seed,
        "phase1_desc":    "Locality-heavy: small hot working set with sequential bursts",
        "phase2_desc":    "Random/bursty: uniform draws from full page range",
    }

    return trace, PHASE1_LEN, metadata
