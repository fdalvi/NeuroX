import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def create_correlation_clusters(
    X, use_abs_correlation=True, clustering_threshold=0.5, method="average"
):
    """
    Args:
        X: #Tokens x #Neurons sized matrix
    Returns:
        independent_neurons: List of neurons (one randomly picked per cluster)
    """
    # Compute correlations
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    print("Correlation matrix size (#neurons x #neurons):", corr.shape)

    # Cluster based on correlations
    if use_abs_correlation:
        dissimilarity = 1 - np.abs(corr)
    else:
        dissimilarity = 1 - corr
    hierarchy = linkage(squareform(dissimilarity), method=method)
    labels = fcluster(hierarchy, clustering_threshold, criterion="distance")
    print("Number of clusters detected: %d" % np.max(labels))

    return labels


def extract_independent_neurons(X, use_abs_correlation=True, clustering_threshold=0.5):
    """
    Args:
        X: #Tokens x #Neurons sized matrix
    Returns:
        independent_neurons: List of neurons (one randomly picked per cluster)
    """
    clusters = create_correlation_clusters(X, use_abs_correlation, clustering_threshold)

    independent_neurons = []

    for i in range(np.min(clusters), np.max(clusters) + 1):
        independent_neurons.append(np.random.choice(np.where(clusters == i)[0]))

    return independent_neurons, clusters


def print_clusters(clusters):
    for i in range(np.min(clusters), np.max(clusters) + 1):
        print(
            "Cluster %05d: %s"
            % (i, " ".join([str(x) for x in np.where(clusters == i)[0]]))
        )

from scipy.spatial.distance import pdist
def scikit_extract_independent_neurons(X, clustering_threshold=0.5):
    c = pdist(X.T, metric='correlation')
    hi = linkage(c, method='average')
    clusters = fcluster(hi, clustering_threshold, criterion='distance')

    independent_neurons = []
    for i in range(np.min(clusters), np.max(clusters) + 1):
        independent_neurons.append(np.random.choice(np.where(clusters == i)[0]))

    return independent_neurons, clusters
