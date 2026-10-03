"""
lru.py — Least Recently Used (LRU) Page Replacement Algorithm

Algorithm logic:
    Keep track of when each page was last accessed.
    On a page fault when frames are full, evict the page whose most
    recent access was furthest in the past (least recently used).

LRU approximates optimal behavior by exploiting temporal locality:
pages used recently are likely to be used again soon.
"""


def run_lru(trace, num_frames):
    """
    Simulate LRU page replacement on a given page-reference trace.

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
    """
    frames = set()
    # Maps page → index of its most recent access
    last_used = {}
    hits = 0
    faults = 0
    per_access = []

    for idx, page in enumerate(trace):
        if page in frames:
            # Page is in memory → hit; update its last-used timestamp
            last_used[page] = idx
            decision = "hit"
            evicted = None
            hits += 1
        else:
            # Page fault
            faults += 1
            if len(frames) < num_frames:
                evicted = None
            else:
                # Evict the page with the smallest last-used index
                # (i.e., the one accessed least recently)
                evicted = min(frames, key=lambda p: last_used[p])
                frames.discard(evicted)
                del last_used[evicted]

            frames.add(page)
            last_used[page] = idx
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
    }
