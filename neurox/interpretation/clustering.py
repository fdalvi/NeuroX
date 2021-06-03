"""Module for clustering analysis.

This module contains functions to perform clustering analysis on neuron
activations.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


def create_correlation_clusters(
    X, use_abs_correlation=True, clustering_threshold=0.5, method="average"
):
    """
    Create clusters based on neuron activation correlation. All neurons in the
    same cluster have "highly correlated" neurons that fire similarly on similar
    inputs.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of size [ NUM_TOKENS x NUM_NEURONS]. Usually the output of
        interpretation.utils.create_tensors
    use_abs_correlation : bool, optional
        Whether to use absolute correlation values. Two neurons that are correlated
        in the opposite direction may represent the same "knowledge" in a large
        neural network.
    clustering_threshold : float, optional
        Hyperparameter for clustering. This is used as the threshold to convert
        hierarchical clusters into flat clusters.

    Returns
    -------
    cluster_labels : list
        List of cluster labels for every neuron

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
    Extract independent neurons from the given set of neurons.

    This method first clusters all of the given neurons with every cluster
    representing similar neurons. A single neuron is then picked randomly from
    every cluster and this forms the final set of independent neurons that is
    returned

    .. seealso::
        `Dalvi, Fahim, et al. "Analyzing redundancy in pretrained transformer models." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. <https://www.aclweb.org/anthology/2020.emnlp-main.398.pdf>`_

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of size [ NUM_TOKENS x NUM_NEURONS]. Usually the output of
        interpretation.utils.create_tensors
    use_abs_correlation : bool, optional
        Whether to use absolute correlation values. Two neurons that are correlated
        in the opposite direction may represent the same "knowledge" in a large
        neural network.
    clustering_threshold : float, optional
        Hyperparameter for clustering. This is used as the threshold to convert
        hierarchical clusters into flat clusters.

    Returns
    -------
    independent_neurons : list
        List of non-redundant indepenent neurons

    """
    clusters = create_correlation_clusters(X, use_abs_correlation, clustering_threshold)

    independent_neurons = []

    for i in range(np.min(clusters), np.max(clusters) + 1):
        independent_neurons.append(np.random.choice(np.where(clusters == i)[0]))

    return independent_neurons, clusters


def print_clusters(cluster_labels):
    """
    Utility function for printing clusters

    Parameters
    ----------
    cluster_labels : list
        List of cluster labels for every neuron. Usually the output of
        ``interpretation.clustering.create_correlation_clusters``.

    """
    for i in range(np.min(cluster_labels), np.max(cluster_labels) + 1):
        print(
            "Cluster %05d: %s"
            % (i, " ".join([str(x) for x in np.where(cluster_labels == i)[0]]))
        )


def scikit_extract_independent_neurons(X, clustering_threshold=0.5):
    """
    Alternative implementation of ``interpretation.clustering.extract_independent_neurons``.

    This is an alternative implementation of the ``extract_independent_neurons``
    function using scikit-learn to create the correlation matrix instead of
    numpy. Should give identical results.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of size [ NUM_TOKENS x NUM_NEURONS]. Usually the output of
        interpretation.utils.create_tensors
    clustering_threshold : float, optional
        Hyperparameter for clustering. This is used as the threshold to convert
        hierarchical clusters into flat clusters.

    Returns
    -------
    independent_neurons : list
        List of non-redundant indepenent neurons
    cluster_labels : list
        List of cluster labels for every neuron

    """
    c = pdist(X.T, metric="correlation")
    hi = linkage(c, method="average")
    clusters = fcluster(hi, clustering_threshold, criterion="distance")

    independent_neurons = []
    for i in range(np.min(clusters), np.max(clusters) + 1):
        independent_neurons.append(np.random.choice(np.where(clusters == i)[0]))

    return independent_neurons, clusters
