"""Module for Probeless method

This module extracts neuron ranking for a label/tag (e.g Verbs) or for an
entire property set (e.g Part of speech) without training any probes.

.. seealso::
        `Antverg, Omer and Belinkov, Yonatan "On The Pitfalls Of Analyzing Idividual Neurons in Language Models." In Proceedings of the 10th International Conference on Learning Representations (ICLR). <https://arxiv.org/abs/2110.07483>`_
"""
import numpy as np
import itertools

def _get_mean_vectors(X_train, y_train):

    embeddings_by_labels = {}
    average_embeddings_by_label = {}
    rankings_per_label = {}

    for words_idx in range(y_train.size):
        label = y_train[words_idx]
        if (label in embeddings_by_labels):
            embeddings_by_labels[label].append(X_train[words_idx])
        else:
            vector = [] 
            vector.append(X_train[words_idx])
            embeddings_by_labels[label] = vector
        
    for k, v in embeddings_by_labels.items():
        average_embeddings_by_label[k] = np.mean(v, axis=0)

    avg_embeddings = list(average_embeddings_by_label.values())
    
    return avg_embeddings, average_embeddings_by_label

def _get_overall_ranking(mean_vectors):
    
    overall = np.zeros_like(mean_vectors[0])
    
    for couple in itertools.combinations(mean_vectors, 2):
        overall += np.abs(couple[0] - couple[1])

    ranking = np.argsort(overall)[::-1].tolist()
    return ranking

def _get_tag_wise_ranking(mean_vectors_by_label, tag):

    qz = mean_vectors_by_label[tag]
    summation = np.abs(np.subtract(qz, qz))

    for c, qzz in mean_vectors_by_label.items():
        
        if (tag != c):
            summation = np.add(summation, np.abs(np.subtract(qz,qzz)))
    
    ranking = np.argsort(summation)[::-1].tolist()
    return summation, ranking 

def get_neuron_ordering(X_train, y_train):
    """
    Returns a list of top neurons w.r.t the overall task e.g. POS

    Parameters
    ----------
    X_train : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the 
        output of ``interpretation.utils.create_tensors``
    y_train : numpy.ndarray
        Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
        token. Usually the output of ``interpretation.utils.create_tensors``.

    Returns
    -------
    ranking : list
        list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.
    """    
    avg_embeddings, average_embeddings_by_label = _get_mean_vectors(X_train, y_train)
    ranking = _get_overall_ranking(avg_embeddings)
    
    return ranking

def get_neuron_ordering_for_tag(X_train, y_train, label2idx, tag):
    """
    Returns a list of top neurons w.r.t a tag e.g. noun

    Parameters
    ----------
    X_train : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the 
        output of ``interpretation.utils.create_tensors``
    y_train : numpy.ndarray
        Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
        token. Usually the output of ``interpretation.utils.create_tensors``.
    label2idx: dict
        Class name to index mapping. Usually returned by
        ``interpretation.utils.create_tensors``.
    tag : string 
        tag for which rankings are extracted

    Returns
    -------
    ranking : list
        list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.

    """
    avg_embeddings, average_embeddings_by_label = _get_mean_vectors(X_train, y_train)
    summation, ranking = _get_tag_wise_ranking(average_embeddings_by_label, label2idx[tag])

    return ranking

def get_neuron_ordering_for_all_tags(X_train, y_train, idx2label):
    """
    Returns a dictionary of tags along with top neurons for each tag
    Returns a list of overall ranking

    Parameters
    ----------
    X_train : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the 
        output of ``interpretation.utils.create_tensors``
    y_train : numpy.ndarray
        Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
        token. Usually the output of ``interpretation.utils.create_tensors``.
    idx2label: dict
        Class index to name mapping. Usually returned by
        ``interpretation.utils.create_tensors``.

    Returns
    -------
    overall_ranking : list
        list of ``NUM_NEURONS`` neuron indices, in decreasing order of importance.
    ranking_per_tag : dict
        Dictionary with top neurons for every class, with the class name as the
        key and list of neurons as the values.
    """
    # TODO: switch to label2idx for consistency
    ranking_per_tag = {}
    avg_embeddings, average_embeddings_by_label = _get_mean_vectors(X_train, y_train)

    overall = np.zeros_like(X_train[0])

    for c, qz in average_embeddings_by_label.items():

        summation, ranking = _get_tag_wise_ranking(average_embeddings_by_label, c)
        overall = np.add(overall, summation)
        ranking_per_tag [idx2label[c]] = ranking

    overall_ranking = np.argsort(overall)[::-1].tolist()

    return overall_ranking, ranking_per_tag
