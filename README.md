# Graph Neural Diffusion as a Learned Spectral Filter — reproduction artefact

Code, figures, tables and raw results for the manuscript *Graph Neural Diffusion
as a Learned Spectral Filter: Closed-Form Propagation and the Cost of
Discretisation*.

**The manuscript is not included in this repository.** It is distributed through
the review system of the journal it is submitted to.

---

## What this repository contains

```
notebook/   GraphSpectralBeltrami.ipynb        the complete pipeline, one file
figures/    Figure_1, Figure_2, Figure_S1–S3     exactly the figures in the paper (PDF and PNG)
tables/     *.tex                               exactly the main tables of the paper
results/    *.csv, *.json                        every raw measurement
scripts/    make_figures.py                      regenerates figures/ from results/
            verify_manuscript_numbers.py         re-checks 31 reported quantities
requirements.lock.txt                            resolved package versions
```

The notebook is self-contained: it acquires the data, runs the experiments,
writes the results, and regenerates the figures and tables. No other script is
needed to reproduce the study.

---

## Reproducing the study

### Quick check (CPU, minutes)

```bash
GB_ARTIFACT=1 jupyter nbconvert --execute --to notebook \
    notebook/GraphSpectralBeltrami.ipynb
```

`GB_ARTIFACT=1` disables Drive mounting, e-mail telemetry and the GPU
requirement, writes to a local directory, and runs the `SMOKE` protocol: two
graphs, two splits, reduced expansion order. It exercises every code path in the
paper and finishes in roughly fifteen minutes on a single CPU core. Use it to
confirm the environment before committing to the full grid.

### Full grid (A100, one session)

Open the notebook and set `RUN_MODE = "MAIN"` in the first cell, or:

```bash
GB_MODE=MAIN jupyter nbconvert --execute --to notebook \
    notebook/GraphSpectralBeltrami.ipynb
```

Six datasets, ten models, ten splits: **1 548 training runs, 12.71 hours on one
NVIDIA A100-SXM4 (40 GB)**. Every phase checks a wall-clock budget and stops
cleanly; a run interrupted by a session limit resumes from `manifest.json`
without repeating completed work.

### Regenerating the figures of the paper

```bash
python scripts/make_figures.py results/ figures/
```

The notebook writes its own diagnostic figures and tables. The files in `figures/`
and `tables/` are the versions used in the paper: they are drawn by
`scripts/make_figures.py` and taken from the manuscript source, from the same raw
results. The discretisation figure and table use the corrected-step-rule pass
`MAIN_e296b125`; nothing in them comes from `results/superseded/`.

### Verifying the reported numbers

```bash
python scripts/verify_manuscript_numbers.py results/
```

Recomputes 31 quantities quoted in the manuscript directly from the raw results
and prints each alongside its reported value. This is the fastest way for a
reviewer to confirm that the tables and the text agree with the measurements.

---

## Environment

Resolved versions are in `requirements.lock.txt`. The floor is
`torch_geometric >= 2.6` (required for `HeterophilousGraphDataset`). The study
was run with PyTorch 2.11 / CUDA 12.8 under Python 3.13.

Training is performed in `float32`; the verification suite runs in `float64`.
Half precision is never used, as it underflows in the term
`1/sqrt(psi_i * psi_j)`.

**On determinism.** Seeds fix initialisation and dropout, cuDNN autotuning is
disabled, and deterministic kernels are selected. Bitwise reproducibility is
*not* claimed: the scatter reductions underlying every graph operator accumulate
with non-deterministic atomics on CUDA, leaving residual floating-point ordering
noise of order 1e-7.

---

## Data

| Tier | Datasets | Source |
|---|---|---|
| Homophilic reference | Cora | PyG `Planetoid` |
| Filtered heterophilic | Chameleon-f, Squirrel-f | downloaded from the release accompanying Platonov et al. (ICLR 2023) |
| Heterophilic benchmark | Roman-empire, Amazon-ratings, Minesweeper | PyG `HeterophilousGraphDataset` |

The unfiltered Chameleon and Squirrel graphs are **not used**. Filtering removes
61 % and 57 % of their nodes as duplicates that leak between the training and
test partitions; the loader prints the reduction at load time.

All downloads are automatic on first run and cached thereafter.

---

## Notes for reviewers

- **`manifest.json` is keyed by `(mode, config hash)`.** The hash covers every
  field that changes the model, including the operator-normalisation switch and
  an explicit code version. Runs produced under one configuration cannot
  contribute to results reported under another.
- **`code_version` carries an opaque build tag from the project's earlier
  development.** It is a field of the hashed configuration, so renaming it
  changes `CONFIG_HASH` and therefore every output filename. It is left as it
  is so that the stored results stay reproducible from this notebook; the
  string itself has no meaning beyond identifying the build.
- **Output directories.** The notebook writes to
  `graphbeltrami_spectral` on Drive, or to `GB_DRIVE_DIR` when that is set.
  Point `GB_DRIVE_DIR` at an existing folder to resume a run that was started
  under a different folder name, otherwise the grid starts again from zero.
- **`results/e3_solver_UNNORMALISED_reference.csv`** is retained from an earlier
  configuration in which the learned operator was not normalised. It is the
  measurement that motivates the normalisation: the certified Chebyshev order ran
  to its cap on every call and one propagation cost 290 s instead of 2 s. It is
  provided so that this design decision can be checked rather than taken on
  trust.
- **Verification is executed before any model is trained.** Seven numerical
  properties (gradient flow, Weyl invariance, permutation equivariance, the
  closed form against a dense reference, spectral estimates, the hand-written
  adjoint, the stability shift) are asserted in double precision; execution halts
  if any fails.
- **Two scale-free residuals are recorded per run** — the relative
  conjugate-gradient residual and the Chebyshev doubling discrepancy — because a
  dense reference exists only for graphs small enough to diagonalise.

---

## Known limitations of the artefact

- The convergence study (E1) builds a dense reference only for graphs with
  `N <= 3000`; larger graphs rely on the run-time residuals.
- The discretization and stability sweeps (E3, E4) run on two datasets and two
  variants.
- The baselines are reimplementations of the published equations inside a common
  backbone, not the authors' code. This isolates the operator, which is the point
  of the study, but the numbers are not the original authors'.

---

## Citation

See `CITATION.cff`. Please cite the published version once it appears.

## License

MIT. See `LICENSE`.
