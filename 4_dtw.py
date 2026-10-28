import os
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy.io
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from helpful_functions import plot_det_curve, print_errors

# ------------------------------
# GLOBAL VARIABLES
# ------------------------------
all_features_np = None  # will be loaded once globally


def init_worker(shared_features):
    """This function runs once when each worker process starts."""
    global all_features_np
    all_features_np = shared_features


def load_signatures(num_users, signatures_per_user, genuine_ids):
    all_features_np = np.empty((num_users, signatures_per_user), dtype=object)

    # iterate over user dimension
    for user in range(1, num_users + 1):
        sig_global_idx = 0  # index 0..15 per user
        # iterate over the session dimension, always 4
        for session in range(1, 5):
            # iterate only over the genuine ids
            for sig in genuine_ids:
                filename = f"u1{user:03d}s{session:04d}_sg{sig:04d}.mat"  # e.g., u0001s0001_sg0001
                filepath = os.path.join("DB", filename)

                data = scipy.io.loadmat(filepath)
                x = data["x"].squeeze()
                y = data["y"].squeeze()
                p = data["p"].squeeze()

                # feature extraction and placing correctly in conclusion array
                feature_matrix = extract_local_features(x, y, p)
                all_features_np[user - 1, sig_global_idx] = feature_matrix

                sig_global_idx += 1  # increment for next signature

    return all_features_np


# ------------------------------
# FEATURE EXTRACTION
# ------------------------------
def extract_local_features(x, y, p):
    dx = np.gradient(x)
    dy = np.gradient(y)
    dp = np.gradient(p)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddp = np.gradient(dp)
    features = np.column_stack([x, y, p, dx, dy, dp, ddx, ddy, ddp])
    # normalization, important for obtaining useful distances
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    return features  # .astype(np.float32)  # float32 to save memory


# ------------------------------
# DTW SCORE
# ------------------------------
def DTW_matcher(test_feat, model_feat):
    """Perform Dynamic Time Warping between two signatures."""
    # fastDTW gives approximate distance in O(T)
    distance, _ = fastdtw(test_feat, model_feat, dist=euclidean)
    K = max(len(test_feat), len(model_feat))
    score = np.exp(-distance / K)
    return score


def Matcher_local(test_signature, model_signatures):
    """Function for performing Dynamic Time Warping over all samples representing a model."""
    scores = [DTW_matcher(test_signature, m) for m in model_signatures]
    return np.mean(scores)


# ------------------------------
# PROCESS A USER CHUNK
# ------------------------------
def process_user_chunk(user_indices, N):
    global all_features_np

    if all_features_np is None:
        raise ValueError("Worker not initialized with data!")

    users = all_features_np.shape[0]
    signatures = all_features_np.shape[1]
    SG = signatures - N  # number of signatures used for testing per user

    # preallocate arrays for efficiency
    GenuineScores_chunk = np.zeros((len(user_indices), SG), dtype=np.float32)
    ImpostorScores_chunk = np.zeros(
        (len(user_indices), SG * (users - 1)), dtype=np.float32
    )

    # loop over each user in this chunk
    for idx, u in enumerate(tqdm(user_indices, desc=f"N={N}")):

        # build the enrollment model using the first N signatures of the user
        Model = [all_features_np[u, i] for i in range(N)]

        # genuine comparisons: test each remaining signature against the model
        for i in range(SG):
            test = all_features_np[u, N + i]
            GenuineScores_chunk[idx, i] = Matcher_local(test, Model)

        # impostor comparisons: test all other users' signatures against the model
        imp_index = 0
        for other_user in range(users):
            if other_user == u:
                continue  # skip self
            for j in range(SG):
                test = all_features_np[other_user, N + j]
                ImpostorScores_chunk[idx, imp_index] = Matcher_local(test, Model)
                imp_index += 1  # move to next position in the impostor array

    return GenuineScores_chunk, ImpostorScores_chunk


# ------------------------------
# PROCESS N SIGNATURES (PARALLEL)
# ------------------------------
def process_N_parallel(N):
    global all_features_np

    if all_features_np is None:
        raise ValueError("Worker not initialized with data!")
    users = all_features_np.shape[0]

    # determine number of CPU cores to use for multiprocessing
    num_cores = min(cpu_count(), 6)  # adjustable
    user_chunks = np.array_split(np.arange(users), num_cores)

    # arguments for each worker process
    args = [(chunk, N) for chunk in user_chunks]

    # preallocate result arrays
    GenuineScores = np.zeros((users, all_features_np.shape[1] - N), dtype=np.float32)
    ImpostorScores = np.zeros(
        (users, (all_features_np.shape[1] - N) * (users - 1)), dtype=np.float32
    )

    # parallel processing of chunks: each worker receives a chunk of users and the global feature array
    with Pool(num_cores, initializer=init_worker, initargs=(all_features_np,)) as pool:
        results = pool.starmap(process_user_chunk, args)

    # combine results from all chunks into the full arrays
    start = 0
    for chunk, (GS_chunk, IS_chunk) in zip(user_chunks, results):
        GenuineScores[chunk, :] = GS_chunk
        ImpostorScores[chunk, :] = IS_chunk

    # save results
    scipy.io.savemat(
        f"GenuineScores_FastDTW_{N}.mat", {f"GenuineScores_FastDTW_{N}": GenuineScores}
    )
    scipy.io.savemat(
        f"ImpostorScores_FastDTW_{N}.mat",
        {f"ImpostorScores_FastDTW_{N}": ImpostorScores},
    )

    return N, GenuineScores, ImpostorScores


# main execution
if __name__ == "__main__":
    Ns = [1, 4, 12]
    num_users = 50
    signatures_per_user = 16
    genuine_ids = [1, 2, 6, 7]

    print("Loading signatures...")
    all_features_np = load_signatures(num_users, signatures_per_user, genuine_ids)
    print(f"Data loaded. Shape: {all_features_np.shape}")

    GenuineScores_list = []
    ImpostorScores_list = []

    # run for all the Ns wanted
    for N in Ns:
        print(f"\nProcessing N={N}")
        _, GS, IS = process_N_parallel(N)
        GenuineScores_list.append(GS)
        ImpostorScores_list.append(IS)

    # obtaining the results
    print_errors(Ns, GenuineScores_list, ImpostorScores_list)
    plot_det_curve(Ns, GenuineScores_list, ImpostorScores_list)
