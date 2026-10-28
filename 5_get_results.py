from pathlib import Path

import scipy.io

from helpful_functions import plot_det_curve, print_errors

Ns = [1, 4, 12]
results_dir = Path(__file__).resolve().parent / "results"

GenuineScores_list = []
ImpostorScores_list = []

for N in Ns:
    # load files
    gs_mat = scipy.io.loadmat(results_dir / f"GenuineScores_FastDTW_{N}.mat")
    is_mat = scipy.io.loadmat(results_dir / f"ImpostorScores_FastDTW_{N}.mat")

    # Extract the actual arrays
    GS = gs_mat[f"GenuineScores_FastDTW_{N}"]
    IS = is_mat[f"ImpostorScores_FastDTW_{N}"]

    GenuineScores_list.append(GS)
    ImpostorScores_list.append(IS)

print_errors(Ns, GenuineScores_list, ImpostorScores_list)
plot_det_curve(Ns, GenuineScores_list, ImpostorScores_list)
