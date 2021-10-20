"""Module for corpus based analysis.

This module contains functions that relate neurons to corpus elements like
words and sentences

"""
from collections import defaultdict

import numpy as np


def get_top_words(tokens, activations, neuron, num_tokens=0):
    """
    Get top activating words for any given neuron.

    This method compares the activations of the given neuron across all tokens,
    and extracts tokens that account for the largest variance for that given
    neuron. It also returns a normalized score for each token, depicting their
    contribution to the overall variance.

    Parameters
    ----------
    tokens : dict
        Dictionary containing atleast one list with the key ``source``. Usually
        returned from ``data.loader.load_data``
    activations : list of numpy.ndarray
        List of *sentence representations*, where each *sentence representation*
        is a numpy matrix of shape
        ``[num tokens in sentence x concatenated representation size]``. Usually
        retured from ``data.loader.load_activations``
    neuron : int
        Index of the neuron relative to ``X``
    num_tokens: int, optional
        Number of top tokens to return. Defaults to 0, which returns all tokens
        with a non-neglible contribution to the variance

    Returns
    -------
    top_neurons : list of tuples
        List of tuples, where each tuple is a (token, score) element

    """
    MIN_THRESHOLD = 0.1  # threshold for "negligible activation score"

    activation_values = [
        sentence_activations[:, neuron] for sentence_activations in activations
    ]
    activation_values = np.concatenate(activation_values)

    tokens = [token for sentence in tokens["source"] for token in sentence]
    mean = np.mean(activation_values)
    std = np.std(activation_values)

    token_wise_scores = np.abs((activation_values - mean) / std)
    type_wise_scores_aggregation = defaultdict(lambda: (0, 0))
    for idx, token in enumerate(tokens):
        curr_sum, curr_count = type_wise_scores_aggregation[token]
        type_wise_scores_aggregation[token] = (
            curr_sum + token_wise_scores[idx],
            curr_count + 1,
        )

    # Normalize by count
    type_wise_scores = [
        (k, v[0] / v[1]) for k, v in type_wise_scores_aggregation.items()
    ]

    # Normalize scores by max
    max_score = max([s for _, s in type_wise_scores])
    type_wise_scores = [(k, v / max_score) for k, v in type_wise_scores]

    # Sort and filter scores
    sorted_types_scores = sorted(type_wise_scores, key=lambda x: -x[1])
    sorted_types_scores = [
        (k, v) for (k, v) in sorted_types_scores if v > MIN_THRESHOLD
    ]

    if num_tokens > 0:
        sorted_types_scores = sorted_types_scores[:num_tokens]

    return sorted_types_scores
