# Bubble_detection_NIMO
Bubble detection by Hough transform and exploration of bubble distribution by BLOX



## Overview

This repository contains the code used to generate all analyses and figures reported in the manuscript.


The framework integrates active learning/AI selection (via `NIMO`) with automated video image analysis (via `Bubblytics`) to optimize bubble counting and distribution fitting experiments. NIMO is a Python library which is available from the followig link. (https://github.com/NIMS-DA/nimo)

## Data
The datasets used in this study (including the recorded videos) are available from Zenodo:

DOI: [10.5281/zenodo.20785492](https://doi.org/10.5281/zenodo.20785492)

## Requirements

Python 3.11

Install packages using

```bash
conda env create -f environment.yml
```


## Repository structure

```text
Bubble_detection_NIMO/
├── experiment/                       # Experiment loop (run from this folder)
│   ├── candidates.csv                # NIMO input: candidate conditions; updated at each run with the measured objectives
│   ├── proposals.csv                 # NIMO output: conditions proposed for the next experiment
│   ├── analysis_output_bubble.py     # Analyzes the videos with Bubblytics; writes the mean and SD of bubble size to candidates.csv and a summary to Res/
│   ├── bubblytics.py                 # Bubble detection from video (Hough transform)
│   ├── main_bubble.ipynb             # Main notebook: run all cells to execute the workflow
│   ├── Res/                          # Created automatically; one summary file (<YYYYMMDDHHMMSS>_res.csv) is added per run
│   └── EXPOutput/                    # NOT in the repository: create it yourself and put the experimental videos in it
│       └── 202601011234/             # One folder per run, named by timestamp (YYYYMMDDHHMM); the newest one is processed
│           ├── Exp_1/                # Folder of the 1st proposal in proposals.csv: put one .avi video here
│           ├── Exp_2/                #   Bubblytics results (including <video name>.csv) are saved
│           └── ...                   #   in the same folder
├── fitting/                          # Bubble-size distribution fitting and model comparison
│   ├── NIMO_fitting_latest_updated.py
│   ├── NIMO_fitting_model_comparison.py
│   ├── compare_alpha_LS_vs_MLE.py
│   └── summary_plots.py
├── environment.yml                   # Conda environment for the experiment workflow
├── environment_fitting.yml           # Exact, pinned environment used for the fitting analysis
├── REPRODUCIBILITY.md                # Detailed reproduction notes for the fitting analysis
├── LICENSE
└── README.md
```

`EXPOutput/` is both the input and the output location of the video analysis.
It is **not included in this repository** (the recorded videos are archived in
the Zenodo data archive listed below) and it is not created automatically.
Before running, create `experiment/EXPOutput/<timestamp>/Exp_<i>/` yourself for
each proposed condition (`i` = 1, 2, ..., in the order of the rows in
`proposals.csv`) and place the recorded video (`.avi`) in it. The results are
saved in the same folder. The notebook processes the newest timestamp folder, so
name run folders in the form `YYYYMMDDHHMM` (digits only) so that they sort
chronologically, and do not put any other files or folders directly inside
`EXPOutput/`. Folder names are case-sensitive on some systems: use `Exp_<i>`
exactly. `Res/` is created automatically if it does not exist. The scripts use
relative paths, so run the notebook from the `experiment` folder.

## Workflow

The analysis has two parts. Part 1 collects bubble-size data with NIMO-driven
experiments; Part 2 fits the bubble-size distributions.

```text
Part 1: Experiment and bubble detection (experiment/)
  candidates.csv --> NIMO (BLOX) proposes the next condition --> proposals.csv
        --> run the experiments and record one .avi video per proposed condition
        --> put each video in EXPOutput/<timestamp>/Exp_<i>/   (folders created by the user)
        --> analysis_output_bubble.py runs bubblytics.py on each video:
              - results saved in the same Exp_<i> folder (<video name>.csv)
              - mean and SD of bubble size over the target frames are written to candidates.csv
              - a summary is added to Res/<YYYYMMDDHHMMSS>_res.csv
        --> (repeat until the experimental budget is used)

Part 2: Distribution fitting (fitting/)
  Bubblytics results CSV of each video (frame number: 1st column, bubble size: 4th column);
  all rows (all frames) are used, with no frame selection
        --> NIMO_fitting_latest_updated.py       -> fit_results.csv (alpha, least squares)
        --> NIMO_fitting_model_comparison.py     -> all_files_model_comparison.csv
                                                    all_files_alpha_uncertainty.csv
        --> compare_alpha_LS_vs_MLE.py           -> agreement between the two alpha estimates
        --> summary_plots.py                     -> summary figures (+ matching .csv for Origin)
```

### Step-by-step

1. **Set up the environment**
   ```bash
   conda env create -f environment.yml
   conda activate bubble-analysis
   ```
   For the fitting analysis, the exact package versions used in the paper are
   listed in `environment_fitting.yml`:
   ```bash
   conda env create -f environment_fitting.yml
   ```
2. **Prepare the experimental data.** In the `experiment` folder, create
   `EXPOutput/`, then `EXPOutput/<timestamp>/` (`YYYYMMDDHHMM`, e.g.
   `202601011234`) and, inside it, one folder `Exp_<i>` per proposed condition (`i` = 1, 2, ...). Place the
   recorded `.avi` video in each `Exp_<i>` folder. The notebook automatically
   uses the newest timestamp folder.
3. **Run the experiment loop.** Move into the `experiment` folder, open
   `main_bubble.ipynb`, and run all cells. The video of each `Exp_<i>` folder
   is analyzed, the results are saved in the same folder, `candidates.csv` is
   updated with the measured values, and a summary file is added to `Res/`.
   After the analysis, the next condition is proposed with the BLOX method of
   NIMO (`nimo.selection(method="BLOX")`), using two objectives (the mean and SD
   of bubble size) and one proposal per cycle; it is written to `proposals.csv`.
   If the message `ErrorCode: error in extract_objectives function` appears,
   check that the folders are named `Exp_<i>` exactly and that each contains an
   `.avi` video.
4. **Fit the distributions.** Run the scripts in `fitting/` in the order
   shown above, e.g. `python fitting/NIMO_fitting_latest_updated.py`.
   Each script opens two folder-selection dialogs (input folder, then output
   folder). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the input and
   output files of each script and for all fixed analysis parameters.

### Analysis parameters of the video analysis

| Parameter | Where it is set | Value used in the paper |
|---|---|---|
| Target frames (`target_flame`) | `main_bubble.ipynb` | frames 20, 22, ..., 80 (31 frames): `[20 + i*2 for i in range(31)]` |
| Number of objectives (`ObjectivesNum`) | `main_bubble.ipynb` | 2 (mean and SD of bubble size) |
| Proposals per cycle (`ProposalsNum`) | `main_bubble.ipynb` | 1 |
| NIMO method | `main_bubble.ipynb` | BLOX |
| Frames used for the distribution fitting | `fitting/` scripts | all frames (no frame selection) |
| Bubble detection parameters (`set_parameters`, `set_detection_parameters`) | `analysis_output_bubble.py` | Defaults |

## Data and code availability

- Data (including the recorded videos): archived on Zenodo, DOI [10.5281/zenodo.20785492](https://doi.org/10.5281/zenodo.20785492).
- Code (version used in the paper): release `v1.0-manuscript`,

## Citation

If you find this code useful for your research, please cite our paper:



## Correspondence

For questions regarding the code, please contact:

[takanabe@chemsys.t.u-tokyo.ac.jp](mailto:takanabe@chemsys.t.u-tokyo.ac.jp)

## License

MIT License
