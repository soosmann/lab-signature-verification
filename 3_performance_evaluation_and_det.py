import numpy as np
import scipy.io

from helpful_functions import plot_det_curve, print_errors


def Matcher(test, model):
    """
    The similarity score between a query/test signature and the enrollment
    signatures (signatures in the database) will be the Euclidean distance
    between feature vectors (vectors with 4 global features). The final
    score will be the average score of the N comparisons (comparison
    between the query/test sample and the N enrollment samples).

    -	Score: is the final score of the comparison.
    -	test: is the feature vector of the query/test signature (size equals to 1x4)
    -	Model: is a matrix containing the N feature vectors of the signatures enrolled in
    the database for the claimed user. Therefore, this matrix contains Nx4 values in which
    N is the number of signatures enrolled for the claimed user (N=1, N=4 and N=12).

    """
    # Euclidean distances between test and each enrolled signature
    distances = np.linalg.norm(model - test, axis=1)

    # Average distance
    avg_distance = np.mean(distances)

    # Avoid division by zero
    if avg_distance == 0:
        return 0

    # Normalize so genuine → higher score
    score = 1 / avg_distance

    return score


# load the corresponding .mat file for genuine signatures
data = scipy.io.loadmat("BiosecurIDparameters_genuine.mat")
BiosecurIDparameters = data["BiosecurIDparameters_genuine"]

# arrays saved in files come with shape (user_count, signature_count, feature_count), so:
users = BiosecurIDparameters.shape[0]
signatures = BiosecurIDparameters.shape[1]

Ns = [1, 4, 12]

GenuineScores_list = []
ImpostorScores_list = []

for N in Ns:

    SG = signatures - N
    print(SG)
    GenuineScores = np.zeros((users, SG))
    ImpostorScores = np.zeros((users, users - 1))

    for u in range(users):

        # Enrollment model
        Model = BiosecurIDparameters[u, 0:N, :]

        # ------------------------
        # Genuine comparisons
        # ------------------------
        for i in range(SG):
            test = BiosecurIDparameters[u, N + i, :]
            # print(test.shape)
            score = Matcher(test, Model)
            GenuineScores[u, i] = score

        # ------------------------
        # Impostor comparisons
        # ------------------------
        imp_index = 0
        for other_user in range(users):
            if other_user != u:

                # First signature of other user
                test = BiosecurIDparameters[other_user, 0, :]

                score = Matcher(test, Model)
                ImpostorScores[u, imp_index] = score
                imp_index += 1

    GenuineScores_list.append(GenuineScores)
    ImpostorScores_list.append(ImpostorScores)

    # Save matrices
    scipy.io.savemat(f"GenuineScores_{N}.mat", {f"GenuineScores_{N}": GenuineScores})

    scipy.io.savemat(f"ImpostorScores_{N}.mat", {f"ImpostorScores_{N}": ImpostorScores})


print_errors(Ns, GenuineScores_list, ImpostorScores_list)
plot_det_curve(Ns, GenuineScores_list, ImpostorScores_list)
