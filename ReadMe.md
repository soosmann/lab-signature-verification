# Online Signature Verification Lab

Biometric verification of handwritten signatures using the BiosecurID dataset. This lab compares global feature matching with Dynamic Time Warping (FastDTW), evaluating how enrollment with **1, 4, or 12 signatures** affects verification performance.

**Authors:** Marcel Hofmann and Carlos Celaya  
**[Read the lab report](docs/Report_Hofmann_Marcel_Celaya_Carlos.pdf)**

## What’s included

| Script | Purpose |
| --- | --- |
| `1_show_signature_representations.py` | Visualize signature trajectories, coordinates, and pressure over time. |
| `2_feature_extraction.py` | Extract duration, pen-up count, pen-down ratio, and average pen-down pressure. |
| `3_performance_evaluation_and_det.py` | Evaluate global features using Euclidean distance matching. |
| `4_dtw.py` | Compare normalized position and pressure signals and their derivatives using FastDTW. |
| `5_get_results.py` | Display results from the FastDTW score files in `results/`. |

The `results/` folder contains precomputed feature matrices and verification scores. The `docs/` folder contains the lab report.

Performance is summarized using **Equal Error Rate (EER)** and **Detection Error Tradeoff (DET)** curves. The experiments use 50 users; verification scores compare genuine signatures with signatures from other users.

## Getting started

Install the Python dependencies:

```bash
python -m pip install numpy scipy matplotlib scikit-learn fastdtw tqdm
```

Run from the repository root. To view the precomputed FastDTW results in `results/` without the raw dataset:

```bash
python 5_get_results.py
```

To reproduce the experiments, place the raw BiosecurID `.mat` files in a `DB/` folder (not included). Scripts expect filenames such as `u1001s0001_sg0003.mat` with `x`, `y`, and `p` arrays.

```bash
python 1_show_signature_representations.py
python 2_feature_extraction.py
python 3_performance_evaluation_and_det.py
python 4_dtw.py
```

Scripts 2–4 currently read and write generated `.mat` files in the repository root, overwriting corresponding existing files. After running script 4, move its `GenuineScores_FastDTW_*.mat` and `ImpostorScores_FastDTW_*.mat` outputs into `results/` before running script 5 to view the new results. Script 5 locates `results/` relative to its own location, so it also works when launched from another directory.

FastDTW computation uses multiprocessing and can take time; use the precomputed score files for a quick overview.
