# Bubble_detection_NIMO
Bubble detection by Hough transform and exploration of bubble distribution by BLOX

# Data-driven exploration of electrode and electrolyte effects on electrochemical bubble distributions

Code accompanying:

Kensei Tsuburaya, Keisuke Obata, Keisuke Nagato, Ryo Tamura, Koji Tsuda, Shoichi Matsuda, Kazuhiro Takanabe, "Data-driven exploration of electrode and electrolyte effects on electrochemical bubble distributions", Journal, Year.

## Overview

This repository contains the code used to generate all analyses and figures reported in the manuscript.

The framework integrates active learning/AI selection (via `NIMO`) with automated video image analysis (via `Bubblytics`) to optimize bubble counting and distribution fitting experiments. NIMO is a Python library which is available from this link. (https://github.com/NIMS-DA/nimo)


## Repository Structure

```text
├── EXPOutput                  # Input folder for video from experiment 
  ├── 202607011024             # Timeslot for analysis. Only newest date is selected
    └── Exp_1                  # Folder for the experiment 
├── analysis_output_bubble.py  # Script for processing video analysis outputs
├── NIMO_fitting.py            # Main optimization and distribution fitting routines
├── bubblytics.py              # Core video/image processing module for bubble counting
├── main_bubble.ipynb          # Jupyter Notebook to execute the entire pipeline
├── input_data/                # Input folder for fitting distribution
├── requirements.txt           # Python dependencies
└── README.md                  # This file

```

## Data Availability

The datasets used in this study are available from Zenodo:

DOI: 10.5281/zenodo.20785492

After downloading the data, place the files in the appropriate directories described below.

## Installation

Clone the repository:

```bash
git clone https://github.com/lab-name/project-name.git
cd project-name
```

Create the analysis environment:

```bash
conda env create -f environment.yml
conda activate project-env
```

## Reproducing the Results

Run the notebooks in the following order:

1. notebooks/01_preprocessing.ipynb
2. notebooks/02_analysis.ipynb
3. notebooks/03_figures.ipynb

The generated figures will be saved in:

```text
results/figures/
```

The generated tables will be saved in:

```text
results/tables/
```
## Citation
If you find this code useful for your research, please cite our paper:


## Correspondence

For questions regarding the code, please contact:

[takanabe@chemsys.t.u-tokyo.ac.jp](mailto:takanabe@chemsys.t.u-tokyo.ac.jp)

## License

MIT License
