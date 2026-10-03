"""
optimal.py — Optimal (Belady's) Page Replacement Algorithm

Algorithm logic:
    On a page fault when frames are full, replace the page whose
    NEXT USE is farthest in the future.
    If a page is not referenced again, it is the ideal eviction candidate
    (its next-use distance is treated as infinity).

This algorithm is provably optimal (fewest possible page faults) but
requires future knowledge, so it cannot be implemented in a real OS.
Its role here is:
    1. A theoretical lower bound for comparison.
    2. A source of ground-truth labels for training the learned policy.

Pre-processing:
    Before simulation begins, we build a lookup table:
        next_use[i] = index of the next access to trace[i] after position i.
    This allows O(1) look-ahead during the simulation.
"""


def _build_next_use(trace):
    """
    Build a list where next_use[i] is the next index after i at which
    trace[i]'s page is accessed again.  If the page is never accessed
    after position i, we store len(trace) (treated as infinity).

    This is computed in a single backward pass over the trace.

    Parameters
    ----------
    trace : list[int]

    Returns
    -------
    list[int]  — same length as trace
    """
    n = len(trace)
    next_use = [n] * n          # default: infinity (never used again)
    last_seen = {}              # page → most recent index seen in backward pass

    for i in range(n - 1, -1, -1):
        page = trace[i]
        if page in last_seen:
            next_use[i] = last_seen[page]
        last_seen[page] = i

    return next_use


def run_optimal(trace, num_frames):
    """
    Simulate Belady's Optimal page replacement on a page-reference trace.

    Parameters
    ----------
    trace : list[int]
        Ordered sequence of page numbers accessed.
    num_frames : int
        Number of physical page frames available.

    Returns
    -------
    dict with keys:
        total_accesses : int
        hits           : int
        faults         : int
        hit_ratio      : float
        fault_ratio    : float
        per_access     : list[dict]  — one entry per access
        next_use       : list[int]   — next-use distances (for label generation)
    """
    next_use = _build_next_use(trace)
    frames = set()
    hits = 0
    faults = 0
    per_access = []

    for idx, page in enumerate(trace):
        if page in frames:
            decision = "hit"
            evicted = None
            hits += 1
        else:
            faults += 1
            if len(frames) < num_frames:
                evicted = None
            else:
                # For each page currently in memory, find how far away
                # its next use is.  Evict the one with the largest gap.
                evicted = max(
                    frames,
                    key=lambda p: next_use[idx] if p == page
                    else _next_use_for_page(p, idx, next_use, trace)
                )
                frames.discard(evicted)

            frames.add(page)
            decision = "fault"

        per_access.append({
            "page": page,
            "decision": decision,
            "evicted": evicted,
            "frames_snapshot": frozenset(frames),
        })

    total = len(trace)
    return {
        "total_accesses": total,
        "hits": hits,
        "faults": faults,
        "hit_ratio": hits / total if total > 0 else 0.0,
        "fault_ratio": faults / total if total > 0 else 0.0,
        "per_access": per_access,
        "next_use": next_use,
    }


def _next_use_for_page(page, current_idx, next_use, trace):
    """
    Return the next index at which `page` is accessed after `current_idx`.
    We scan next_use[] entries that correspond to occurrences of `page`.

    Because next_use[i] gives the next use of trace[i] starting from i,
    we need the first occurrence of `page` at or after current_idx and then
    follow the chain.

    A simpler approach: search forward from current_idx + 1.
    The trace is at most a few thousand elements, so this is acceptable.
    """
    n = len(trace)
    for j in range(current_idx + 1, n):
        if trace[j] == page:
            return j
    return n   # never used again → infinity
