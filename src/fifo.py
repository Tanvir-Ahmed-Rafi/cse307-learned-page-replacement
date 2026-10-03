"""
fifo.py — First-In, First-Out (FIFO) Page Replacement Algorithm

Algorithm logic:
    Maintain a queue of pages currently in memory.
    On a page fault when frames are full, evict the oldest page
    (the one that entered the frames first).

This is the simplest page replacement algorithm.
It is easy to implement but ignores recency of use.
"""

from collections import deque


def run_fifo(trace, num_frames):
    """
    Simulate FIFO page replacement on a given page-reference trace.

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
    frames = set()          # set for O(1) membership check
    queue = deque()         # tracks insertion order (oldest first)
    hits = 0
    faults = 0
    per_access = []

    for page in trace:
        if page in frames:
            # Page already in memory → hit
            decision = "hit"
            evicted = None
            hits += 1
        else:
            # Page not in memory → fault
            faults += 1
            if len(frames) < num_frames:
                # There is a free frame; just bring the page in
                evicted = None
            else:
                # Frames full: evict the page that arrived first
                evicted = queue.popleft()
                frames.discard(evicted)

            frames.add(page)
            queue.append(page)
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
