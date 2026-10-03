# Learning-Augmented Page Replacement Under Workload Shifts

**Course:** CSE-307 — Operating Systems  
**Track:** Track 1 — Learned Page Replacement (Memory Management)  
**Student Name:** Tanvir Ahmed Rafi  
**Student ID:** 202414025  
**Section:** A  
**Batch:** CSE-24  
**GitHub Repository:** [https://github.com/Tanvir-Ahmed-Rafi/cse307-learned-page-replacement](https://github.com/Tanvir-Ahmed-Rafi/cse307-learned-page-replacement)  

---

## 1. Problem Statement

Page replacement algorithms decide which memory page to evict when a page fault
occurs and all page frames are occupied.  Classical algorithms such as FIFO and
LRU are designed for *stationary* workloads.  Real applications, however, switch
between phases with very different memory-access patterns — a locality-heavy
startup phase, followed by a random streaming phase, for example.  This project
investigates whether a lightweight machine-learning policy (Decision Tree) can
outperform classical algorithms across both phases of such a workload shift.

---

## 2. Selected Track

**Track 1 — Learned Page Replacement (Memory Management)**

---

## 3. Objective

- Implement FIFO, LRU, and Belady's Optimal page replacement from scratch.
- Design a reproducible two-phase synthetic trace with a deliberate workload shift.
- Train a Decision Tree classifier to predict the best eviction candidate using
  only *causal* features (no future information at inference time).
- Compare all four policies on the same trace, before and after the shift.
- Report actual measured results; no fabricated numbers.

---

## 4. Algorithms Implemented

| Algorithm | File | Description |
|-----------|------|-------------|
| FIFO | `src/fifo.py` | Evicts the page that has been in memory the longest. |
| LRU | `src/lru.py` | Evicts the page not used for the longest time. |
| Optimal (Belady's) | `src/optimal.py` | Evicts the page whose next use is farthest in the future. Requires future knowledge — used as a theoretical lower bound and label source. |
| Learned Policy | `src/learned_policy.py` | Decision Tree trained on Phase-1 data; predicts eviction target using recency, frequency, and recency_ratio. |

---

## 5. Learned Component

**Model:** `DecisionTreeClassifier` (scikit-learn), `max_depth=4`, `min_samples_leaf=5`

**Features (all causal — computed only from past accesses):**

| Feature | Description |
|---------|-------------|
| `recency` | Steps since the page was last accessed (large = stale) |
| `frequency` | Total number of times the page has been accessed so far |
| `recency_ratio` | `recency / (current_step + 1)` — normalised recency |

**Target label:** `1` if this candidate is the page Optimal (Belady's) would evict;
`0` otherwise.  Future knowledge is used **only** to generate labels, never as a
model input feature.

**Training/Evaluation split:** 75 % train / 25 % test, stratified.

**Measured model performance (held-out test set):**

| Metric | Value |
|--------|-------|
| Accuracy | 0.8692 |
| Precision | 1.0000 |
| Recall | 0.3333 |
| F1 | 0.5000 |
| Train samples | 318 |
| Test samples | 107 |

---

## 6. Workload Design

Implemented in `src/workload.py`.

| Parameter | Value |
|-----------|-------|
| Total trace length | 1000 |
| Phase 1 length (indices 0–499) | 500 |
| Phase 2 length (indices 500–999) | 500 |
| Page range | 0–24 (25 pages) |
| Hot working set (Phase 1) | Pages 0–5 (6 pages) |
| Sequential burst length (Phase 1) | 4 consecutive pages |
| Background-access probability (Phase 1) | 10 % |
| Random seed | 42 |

---

## 7. Workload Shift

**Phase 1 (locality-heavy):** Accesses are concentrated on a 6-page hot working
set (pages 0–5).  Approximately 40 % of accesses form short sequential bursts
(e.g., pages 2→3→4→5), mimicking code/data spatial locality.  10 % of accesses
go outside the hot set as background noise.

**Workload shift:** At index 500, the access pattern changes abruptly.

**Phase 2 (random/bursty):** Page numbers are drawn uniformly at random from the
full 25-page range.  No working-set discipline; every page is equally likely.
This models a context-switch to a streaming or scanning workload.

---

## 8. Experimental Setup

- **Same trace** used for all four algorithms (fair comparison).
- **Frame counts tested:** 3, 4, 5, 6.
- Statistics collected both **overall** and **per phase**.
- All numbers come directly from running `run_experiment.py` — nothing fabricated.

---

## 9. Installation

```bash
# Clone / navigate into the project directory, then:
pip3 install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib` (see `requirements.txt`).

---

## 10. How to Run

```bash
python3 run_experiment.py
```

This single command:
1. Generates the synthetic trace (seed = 42)
2. Runs FIFO, LRU, Optimal for all frame counts
3. Collects training samples and trains the Decision Tree
4. Runs the Learned policy
5. Writes CSV files to `results/`
6. Generates figures to `results/figures/`
7. Prints a summary table to the terminal

---

## 11. Output Files

```
results/
├── raw_results.csv       — Overall stats per (algorithm, frame_count)
├── phase_results.csv     — Per-phase stats per (algorithm, frame_count, phase)
├── model_metrics.csv     — Decision Tree evaluation metrics
└── figures/
    ├── workload_trace.png        — Scatter plot of trace + sliding-window working-set size
    ├── hit_ratio_comparison.png  — Hit ratio vs. frames for all 4 algorithms
    ├── page_fault_comparison.png — Page faults vs. frames for all 4 algorithms
    └── phase_comparison.png      — Side-by-side bar chart: Phase 1 vs Phase 2 hit ratios (frames=4)
```

---

## 12. Results Summary

### Overall (all 1000 accesses)

| Algorithm | Frames=3 Faults | Frames=4 Faults | Frames=5 Faults | Frames=6 Faults |
|-----------|:-:|:-:|:-:|:-:|
| FIFO      | 807 | 710 | 599 | 509 |
| LRU       | 810 | 718 | 612 | 476 |
| Optimal   | 579 | 460 | 368 | 296 |
| Learned   | 757 | 657 | 539 | 443 |

### Phase breakdown (frames = 4)

| Algorithm | Phase 1 Hit Ratio | Phase 2 Hit Ratio | Drop |
|-----------|:-:|:-:|:-:|
| FIFO      | 0.432 | 0.148 | −0.284 |
| LRU       | 0.412 | 0.152 | −0.260 |
| Optimal   | 0.698 | 0.382 | −0.316 |
| Learned   | 0.546 | 0.140 | −0.406 |

All algorithms degrade significantly after the workload shift.  Optimal retains
the highest hit ratio in both phases.  The Learned policy outperforms FIFO and
LRU in Phase 1 (where it was trained) but degrades more sharply in Phase 2
(where the distribution it learned no longer applies).

---

## 13. Limitations

- **Learned policy trained on Phase 1 only:** The model is not re-trained or
  updated during Phase 2, so it cannot adapt online.  An online learning variant
  would be an interesting extension.
- **Small trace / page range:** 1000 accesses and 25 pages are sufficient for
  classroom demonstration but do not reflect production workload scales.
- **Imbalanced labels:** Only one page is the optimal eviction target per
  eviction event.  The model achieves high precision (1.00) but low recall (0.33),
  meaning it misses many optimal evictions and falls back to non-optimal choices.
- **No Belady anomaly testing:** FIFO is susceptible to Belady's anomaly (more
  frames → more faults), but this trace does not specifically exercise that case.
- **Decision Tree cannot generalise beyond training distribution:** Its features
  are simple and the model makes rule-based splits that may not generalise to
  very different workloads.

---

## 14. AI Assistance Disclosure

AI coding assistants were used for implementation guidance, debugging assistance,
and code organisation.  The experimental design, execution, generated results,
interpretation, and final analysis were reviewed and completed by the student.

---

## 15. File Descriptions

| File | Purpose |
|------|---------|
| `run_experiment.py` | Single entry point — runs everything end-to-end |
| `src/fifo.py` | FIFO page replacement implementation |
| `src/lru.py` | LRU page replacement implementation |
| `src/optimal.py` | Belady's Optimal implementation with backward-pass next-use precomputation |
| `src/workload.py` | Two-phase synthetic trace generator |
| `src/features.py` | Feature engineering and training-sample collection |
| `src/learned_policy.py` | Decision Tree training, evaluation, and inference |
| `src/evaluation.py` | Phase-split statistics and result collection helpers |
| `results/` | All generated CSV files and figures |
| `report/term_paper.tex` | LaTeX term paper |
