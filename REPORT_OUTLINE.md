# Checkpoint 2 Report Outline: 3 pages max, all-inclusive

Constraints from the assignment: report.pdf, **no more than 3 pages total**
including figures, tables, references, and contributions; no cover page or
appendices; ≥10-point text, 0.75-inch margins. Running instructions belong
in the README, not the report. GitHub repo URL + commit hash must appear
in the report (and README).

## Page 1: Implementation and data

- **§1 Implementation** (~40% of p.1)
  - Pipeline recap in 3–4 lines (annotations → frozen CoTracker3 →
    TTO on qp_emb → endpoints → phases), pipeline figure (paper Fig. 1
    or our overlay still).
  - What changed vs the Checkpoint 1 plan, stated plainly (e.g. TTO
    settings actually used: paper's 1,000 steps / lr 1e-3→1e-5 vs code
    defaults 500 / 1e-4, say which and why).
  - Reused components: PosePAL repo (commit 63572cd), CoTracker3
    weights; what we wrote ourselves.
- **§2 Data and splits** (~60% of p.1)
  - Sources: stallion clips (IDs, horses, lengths) + paper datasets.
  - Annotation protocol: frame 0 + every ~10th frame corrections;
    blind scoring frames 5/15/25… kept separate (no leakage).
  - Split: by horse (or whole clip), counts table, split manifest name;
    ≥100 reference frames across ≥5 clips, state what was achieved.

## Page 2: Evaluation and results

- **§3 Evaluation protocol** (~30% of p.2)
  - Metrics: δ_avg (1/2/4/8/16 px @256×256), J, **J_masked (declared
    primary metric, per Checkpoint 1)**, endpoint errors, phase
    macro-F1. Declared success target, stated before results.
  - Model choices frozen before final testing, say so explicitly.
- **§4 Results** (~70% of p.2), tables, not prose:
  - Tracking: frozen CoTracker3 vs adapted PosePAL (δ_avg, J, J_masked)
    on held-out frames; improvement experiment (uncertainty-selected
    corrections, same budget) as a third row/column.
  - Endpoints: per-endpoint error vs manual reference, coverage
    (frames missing due to occlusion).
  - Phases: rule-based vs random forest (pose-only vs temporal),
    macro-F1 per phase.
  - One overlay still from the demo video.

## Page 3: Limitations, reproducibility, contributions

- **§5 Limitations and honest assessment** (~50% of p.3)
  - Targets met/missed and the experiments run in response (required
    by the assignment, unmet targets reported honestly).
  - Image-plane only; occlusion gaps in mount/thrust; annotation
    budget; clip count.
- **§6 Reproducibility** (~20%): repo URL, commit hash, one-line run
  pointer to README, hardware + runtime.
- **References** (~15%): short list (PosePAL, CoTracker3, + datasets).
- **Contributions** (~15%): the four-row table from the README.

## Assembly notes

- Owner: Kelly (assembly + final pass), sections drafted by the owner
  of each part (Stephen §1 runs, Sam §2/§4 endpoints, Jim §3/§4 scores).
- Freeze all numbers by Thu Nov 19 so the report, slides, and README
  tell the same story; the ZIP is due Fri Nov 20, 11:59 PM ET.
