# Reproducibility: bubble-size-distribution fitting and model comparison

This document describes the fitting/model-comparison code added under
`src/fitting/`, the exact environment used to run it, and the step-by-step
workflow needed to reproduce the results reported in the manuscript
(Methods §2.3–2.8). It is written in response to a reviewer request for
"the exact software version or repository commit, analysis parameters, and a
short reproducible workflow" for the publicly available code and data.

## 1. Scripts added (`src/fitting/`)

| Script | Purpose | Corresponds to |
|---|---|---|
| `NIMO_fitting_latest_updated.py` | Least-squares (histogram SSE) fit of the Rayleigh–Normal mixture model. Produces the α values used throughout the main text/figures. | Methods §2.3.1 |
| `NIMO_fitting_model_comparison.py` | MLE fit of all four candidate models (mixture, Gamma, Lognormal, Weibull); AIC/BIC/ΔAIC/ΔBIC/Akaike weights; KS goodness-of-fit test; 5-fold cross-validation (incl. ΔCV); non-parametric bootstrap uncertainty for α. | Methods §2.3.2, §2.5–2.8.1 |
| `compare_alpha_LS_vs_MLE.py` | Agreement analysis between the least-squares α and the MLE α (Pearson/Spearman correlation, Bland–Altman). | Methods §2.8.2 |
| `summary_plots.py` | Aggregates the per-file outputs of `NIMO_fitting_model_comparison.py` across all experimental conditions into summary figures (AIC/BIC, ΔAIC, KS, in-sample vs. CV, ΔCV, α forest plot, best-model frequency). | Figures S_ (SI) |

Each script that produces a figure also writes a same-named `.csv` file with
the plotted values, so that figures can be reproduced directly (e.g., in
OriginLab) without re-running the analysis.

*(Optional, not used in the manuscript's reported results: `compare_single_distribution_fits.py`,
an independent Rayleigh-only vs. Normal-only fit comparison with a Vuong
(1989) test, used only as an internal robustness check during development.
Include only if you want to document that check publicly.)*

## 2. Exact software version

The version constraints in the existing `environment.yml` (`numpy`, `pandas`,
`scipy`, `matplotlib`, unpinned) are not sufficient to guarantee exact
reproducibility, since results (particularly the multi-start optimizer,
`scipy.optimize.minimize` with SLSQP) can in principle differ across SciPy
versions. To fix this:

1. In the Anaconda environment actually used to run the analysis, export the
   exact package versions:

   ```bash
   conda activate bubble-analysis      # or your environment's name
   conda env export --no-builds > environment_fitting.yml
   ```

2. Commit `environment_fitting.yml` alongside the scripts (see §4). This file
   pins every package to the exact version used, unlike the existing
   `environment.yml`.

3. Report these same version numbers in the manuscript's Methods
   (§2.9 "Implementation and reproducibility" in `methods_draft_v2.md`), e.g.:

   > All analyses were implemented in Python 3.11.x using numpy X.X.X,
   > pandas X.X.X, scipy X.X.X, and matplotlib X.X.X (see
   > `environment_fitting.yml` in the code repository for the complete,
   > pinned environment).

   To fill in the X.X.X values, run the following in the same environment
   and copy the printed numbers:

   ```bash
   python -c "import sys, numpy, pandas, scipy, matplotlib; \
   print('python', sys.version.split()[0]); \
   print('numpy', numpy.__version__); \
   print('pandas', pandas.__version__); \
   print('scipy', scipy.__version__); \
   print('matplotlib', matplotlib.__version__)"
   ```

## 3. Analysis parameters (fixed, used identically across all 20 conditions)

| Parameter | Value |
|---|---|
| Grid-search initial values for μ_n | sample mean + {−30,−20,−10,0,10,20,30} |
| Grid-search initial values for loc_r | {0, 10, 20, 30, 40, 50} |
| Total initial-value combinations | 7 × 6 = 42 |
| Optimizer | `scipy.optimize.minimize`, method="SLSQP" |
| Upper bound on loc_r | 20th percentile of each dataset |
| Lower bound on scale_r, σ_n | max(3.0, 0.10 × SD(x)) per dataset |
| Ordering constraint | μ_n − loc_r ≥ 0 |
| Non-negativity constraint | μ_n − 3σ_n ≥ 0 |
| Cross-validation folds | k = 5 |
| Bootstrap resamples | 200 |
| Random seed (CV split and bootstrap) | 42 |

These values are hardcoded as named constants at the top of
`NIMO_fitting_model_comparison.py` (and mirrored in
`NIMO_fitting_latest_updated.py` for the boundary conditions), so a reader
can confirm them directly in the code rather than only in the manuscript
text.

## 4. Step-by-step reproducible workflow

```text
data/<condition>.csv                  (raw NIMO output, one file per experimental condition)
        |
        v
NIMO_fitting_latest_updated.py        -> fit_results.csv                (LS alpha, all conditions)
                                       -> <condition>_pdf_values.csv
                                       -> <condition>_fit.png
        |
        v
NIMO_fitting_model_comparison.py      -> all_files_model_comparison.csv (AIC/BIC/KS/CV, all conditions x 4 models)
                                       -> all_files_alpha_uncertainty.csv (bootstrap summary, all conditions)
                                       -> <condition>_model_comparison.png/.csv
                                       -> <condition>_mixture_decomposition.png/.csv
                                       -> <condition>_pdf_values_all_models.csv
        |
        v
compare_alpha_LS_vs_MLE.py            -> alpha_LS_vs_MLE_merged_data.csv
   (reads fit_results.csv +              alpha_LS_vs_MLE_summary_stats.csv
    all_files_alpha_uncertainty.csv)   -> scatter / Bland-Altman figures + .csv
        |
        v
summary_plots.py                      -> summary_*.png + matching .csv (SI figures)
   (reads all_files_model_comparison.csv,
    all_files_alpha_uncertainty.csv)
```

Each script is run as `python <script_name>.py`; each opens two folder-picker
dialogs (input folder, output folder) rather than taking command-line
arguments, so no additional configuration is required beyond selecting the
correct folders in order.

## 5. Citing an exact, permanent version in the manuscript

A plain link to the repository's default branch (`main`) is not sufficient
for reproducibility, because `main` can change after publication. Two
complementary steps fix this:

1. **Tag the exact commit used for the submitted manuscript:**

   ```bash
   git tag -a v1.0-manuscript -m "Code version used for [Journal] submission"
   git push origin v1.0-manuscript
   ```

   Cite this as: "Analysis code (commit tag `v1.0-manuscript`) is available
   at https://github.com/Catalysis-for-Energy-Conversion/Bubble_detection_NIMO".

2. **(Recommended) Archive the tagged release on Zenodo to obtain a DOI:**
   - Go to https://zenodo.org, sign in with your GitHub account.
   - In your Zenodo account settings ("GitHub" tab), toggle the repository on.
   - On GitHub, create a Release from the `v1.0-manuscript` tag (Releases →
     "Draft a new release"). This triggers Zenodo to automatically archive a
     snapshot and mint a DOI.
   - Cite the DOI in the manuscript's code/data availability statement, e.g.:
     "Analysis code is archived at Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX".

   A DOI is preferred by most journals/reviewers over a bare GitHub link
   because it points to an immutable snapshot even if the GitHub repository
   is later deleted, renamed, or force-pushed.
