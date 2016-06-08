import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from operator import itemgetter


## compare the clusters similarity to the hyades boolean vector and return best cluster label and its accuracy
def compute_similarity_to_true_hyades(clusters, truth_vec):
    clust_labels = np.unique(clusters)

    best_cluster = -1
    best_cluster_acc = 0
    for c in clust_labels:
        clust = clusters == int(c)
        hits = np.logical_and(clust, truth_vec).sum()
        # print(hits, truth_vec.sum())
        acc = float(hits) / truth_vec.sum()
        if acc > best_cluster_acc:
            best_cluster = c
            best_cluster_acc = acc

    return best_cluster, best_cluster_acc


def kmeans(vectors: list, num_rows, k):
    matrix = []
    ## num_rows X len(vectors)
    for s in range(num_rows):
        row = []
        for v in vectors:
            row.append(v[s])
        matrix.append(np.array(row))

    matrix = np.array(matrix)

    clustering = KMeans(n_clusters=k, init='k-means++', n_init=10)
    clustering.fit(matrix)
    return clustering.predict(matrix)


# vectors = [ ra, dec, pm_ra, pm_dec, parallax, distance, gal_long, gal_lat ]
def find_optimal_kmeans(max_k, vectors, true_hyades_vec):
    ## table with max_k rows and 9 columns (1 per criterion)
    kmeans_acc = {i : [None] * max_k for i in range(9)}

    num_rows = vectors[0].shape[0]
    ## for each k in range(0, max_k + 1)
    for k in range(1, max_k + 1):
        ## for each selection criterion
        for i in range(9):
            ## store accuracy in vector where index + 1 -> k
            if i == 0:  ## ra, dec
                clusters = kmeans([vectors[0], vectors[1]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)
            elif i == 1:  ## parallax
                clusters = kmeans([vectors[4]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)
            elif i == 2:  ## distance
                clusters = kmeans([vectors[5]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)
            elif i == 3:  ## galactic long/lat
                clusters = kmeans([vectors[6], vectors[7]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)
            elif i == 4:  ## proper motions
                clusters = kmeans([vectors[2], vectors[3]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)
            elif i == 5:  ## ra, dec, distance
                clusters = kmeans([vectors[0], vectors[1], vectors[5]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)
            elif i == 6:  ## ra, dec, parallax
                clusters = kmeans([vectors[0], vectors[1], vectors[4]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)
            elif i == 7:  ## distance, long, lat
                clusters = kmeans([vectors[5], vectors[6], vectors[7]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)
            else:  ## distance, proper motions
                clusters = kmeans([vectors[5], vectors[2], vectors[3]], num_rows, k)
                label, accuracy = compute_similarity_to_true_hyades(clusters, true_hyades_vec)
                assert k == np.unique(clusters).shape[0]
                kmeans_acc[i][k - 1] = (accuracy, clusters, k, label)

    ## For each criterion - sort tuples on first element - tuple contains accuracy and cluster label to use for mask
    optimal = [None] * 9
    for criterion in kmeans_acc:
        ## ignore k = 1
        best = sorted(kmeans_acc[criterion],key=itemgetter(0))[1]
        print(best)
        optimal[int(criterion)] = best
    return optimal
