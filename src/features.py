"""
features.py — Feature Engineering for the Learned Page Replacement Policy

Feature design
==============
At every eviction decision point (a page fault with full frames), we have
a set of candidate pages currently in memory.  For each candidate we
compute three simple, *causally safe* features — features that rely only
on information available UP TO the current access, never on future accesses.

Feature 1: recency
    How many steps ago was this page last accessed?
    A large value means the page has not been used for a long time.
    (Higher recency → better eviction candidate.)

Feature 2: frequency
    How many times has this page been accessed in the entire history
    seen so far?
    (Lower frequency → better eviction candidate.)

Feature 3: recency_ratio
    recency / (step_index + 1)
    Normalises recency against the current position so the model can
    generalise across different stages of the trace.

Target label (for training only)
---------------------------------
    1 if this candidate is the page that Optimal (Belady's) would evict.
    0 otherwise.

Future information is ONLY used to generate the training label.
It is NEVER given as an input feature.

Sample generation
-----------------
For each eviction point in the trace:
    - Identify all pages currently in frames.
    - Compute the three features above for each candidate page.
    - Using Belady's next-use table (built from the full trace), find
      which candidate has the farthest next use → that is the optimal target.
    - Store one training row per candidate, labelled 1/0.

This creates an imbalanced dataset (one 1 per eviction, rest 0s), which
is acceptable for a lightweight binary classifier.  We do NOT oversample
or upsample because the natural imbalance reflects the real decision space.
"""

import numpy as np


def compute_candidate_features(candidate_pages, current_idx, access_history):
    """
    Compute features for a list of candidate pages at a given time step.

    Parameters
    ----------
    candidate_pages : iterable[int]
        Pages currently in memory (eviction candidates).
    current_idx : int
        Current position in the trace (0-based).
    access_history : dict[int, list[int]]
        Maps each page to a list of indices at which it has been accessed
        so far (up to and including current_idx).

    Returns
    -------
    list[dict]  — one feature dict per candidate page
    """
    rows = []
    for page in candidate_pages:
        history = access_history.get(page, [])

        # Feature 1: recency — steps since last access
        if history:
            last_access = history[-1]
            recency = current_idx - last_access
        else:
            recency = current_idx + 1   # never accessed (treat as very old)

        # Feature 2: frequency — total accesses so far
        frequency = len(history)

        # Feature 3: normalised recency
        recency_ratio = recency / (current_idx + 1)

        rows.append({
            "page":          page,
            "recency":       recency,
            "frequency":     frequency,
            "recency_ratio": recency_ratio,
        })
    return rows


def build_training_samples(trace, num_frames, next_use_table):
    """
    Replay the trace and collect labelled training samples at every
    eviction decision point.

    Uses Belady's next_use_table to determine the optimal eviction target
    (label = 1) at each decision.  All other candidates get label = 0.

    IMPORTANT: next_use_table is used ONLY to compute the label, never
    as a model input feature.

    Parameters
    ----------
    trace : list[int]
    num_frames : int
    next_use_table : list[int]
        next_use_table[i] = next index after i where trace[i]'s page
        is accessed.  len(trace) means "never".

    Returns
    -------
    X : list[list[float]]   — feature vectors  [recency, frequency, recency_ratio]
    y : list[int]           — binary labels (1 = optimal eviction target)
    """
    frames = set()
    access_history = {}   # page → list of past access indices

    X = []
    y = []

    for idx, page in enumerate(trace):
        # Update access history (before computing features, so we include
        # the current access in the frequency count for pages already in memory,
        # but NOT for the incoming page during eviction deliberation).
        # We update the history AFTER the eviction decision to keep features
        # strictly based on information before the current fault.
        # → Actually, for hits the current access should count toward history.
        # Strategy: update history only after the eviction decision is made.

        if page in frames:
            # Hit: no eviction needed; just update history
            access_history.setdefault(page, []).append(idx)
        else:
            # Fault: eviction may be needed
            if len(frames) == num_frames:
                # Need to evict: compute features for each candidate
                candidate_features = compute_candidate_features(
                    frames, idx, access_history
                )

                # Determine optimal eviction target via Belady's next-use
                def next_use_of(p):
                    """Next use of page p after the current index idx."""
                    for j in range(idx + 1, len(trace)):
                        if trace[j] == p:
                            return j
                    return len(trace)   # never used again

                best_candidate = max(frames, key=next_use_of)

                for cf in candidate_features:
                    feat_vector = [
                        cf["recency"],
                        cf["frequency"],
                        cf["recency_ratio"],
                    ]
                    label = 1 if cf["page"] == best_candidate else 0
                    X.append(feat_vector)
                    y.append(label)

                # Evict the optimal page (mirrors what Optimal would do)
                frames.discard(best_candidate)

            frames.add(page)
            access_history.setdefault(page, []).append(idx)

    return X, y
