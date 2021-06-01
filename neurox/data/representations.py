"""Utility functions to manage representations.

This module contains functions that will help in managing extracted
representations, specifically on sub-word based data.
"""
import numpy as np
from tqdm import tqdm


def bpe_get_avg_activations(tokens, activations):
    """Aggregates activations by averaging assuming BPE-based tokenization.

    Given loaded tokens data and activations, this function aggeregates
    activations based on tokenized text. BPE based tokenization is assumed,
    with every non-terminal subword ending with "@@". The activations are
    aggregated by averaging over subwords.

    .. warning::
        This function is deprecated and will be removed in future versions.

    Parameters
    ----------
    tokens : dict
        Dictionary containing three lists, ``source``, ``source_aux`` and
        ``target``. Usually the output of ``data.loader.load_aux_data``.
    activations : list of numpy.ndarray
        Activations returned from ``loader.load_activations``.

    Returns
    -------
    activations : list of numpy.ndarray
        Subword aggregated activations corresponding to one per actual token
        found in the untokenized text.

    """
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in range(0, len(tokens["source_aux"])):
        sourceIndex = 0
        thisBPE = ""
        source = tokens["source"][i]
        source_aux = tokens["source_aux"][i]
        num_words = len(source)
        new_activations = np.zeros((num_words, num_neurons))

        word_boundaries = []

        for j in range(0, len(tokens["source_aux"][i])):
            currSourceWord = tokens["source"][i][sourceIndex]
            thisBPE = thisBPE + tokens["source_aux"][i][j]

            if thisBPE != currSourceWord:
                thisBPE = thisBPE[:-2]
            else:
                word_boundaries.append(j)
                sourceIndex = sourceIndex + 1
                thisBPE = ""

        assert len(word_boundaries) == num_words

        prev_idx = 0
        for word_idx, boundary in enumerate(word_boundaries):
            avg_vector = np.average(activations[i][prev_idx : boundary + 1, :], axis=0)
            new_activations[word_idx, :] = avg_vector
            prev_idx = boundary + 1

        all_activations.append(new_activations)

    return all_activations


def bpe_get_last_activations(tokens, activations, is_brnn=True):
    """Aggregates activations by picking the last subword assuming BPE-based tokenization.

    Given loaded tokens data and activations, this function aggeregates
    activations based on tokenized text. BPE based tokenization is assumed,
    with every non-terminal subword ending with "@@". The activations are
    aggregated by picking the last subword for any given word.

    .. warning::
        This function is deprecated and will be removed in future versions.

    Parameters
    ----------
    tokens : dict
        Dictionary containing three lists, ``source``, ``source_aux`` and
        ``target``. Usually the output of ``data.loader.load_aux_data``.
    activations : list of numpy.ndarray
        Activations returned from ``loader.load_activations``.
    is_brnn : bool, optional
        Whether the model from which activations were extracted was bidirectional.
        Only applies for RNN models.

    Returns
    -------
    activations : list of numpy.ndarray
        Subword aggregated activations corresponding to one per actual token
        found in the untokenized text.

    """
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in range(0, len(tokens["source_aux"])):
        sourceIndex = 0
        thisBPE = ""
        source = tokens["source"][i]
        source_aux = tokens["source_aux"][i]
        num_words = len(source)
        new_activations = np.zeros((num_words, num_neurons))

        word_boundaries = []

        for j in range(0, len(tokens["source_aux"][i])):
            currSourceWord = tokens["source"][i][sourceIndex]
            thisBPE = thisBPE + tokens["source_aux"][i][j]

            if thisBPE != currSourceWord:
                thisBPE = thisBPE[:-2]
            else:
                word_boundaries.append(j)
                sourceIndex = sourceIndex + 1
                thisBPE = ""

        assert len(word_boundaries) == num_words

        rnn_boundary = int(num_neurons / 2)
        if not is_brnn:
            rnn_boundary = num_neurons

        prev_idx = 0
        for word_idx, boundary in enumerate(word_boundaries):
            # 0 - num_neurons/2: Forward
            # num_neurons/2 - : Backward
            new_activations[word_idx, :rnn_boundary] = activations[i][
                boundary, :rnn_boundary
            ]
            if is_brnn:
                new_activations[word_idx, rnn_boundary:] = activations[i][
                    prev_idx, rnn_boundary:
                ]
            prev_idx = boundary + 1

        all_activations.append(new_activations)

    return all_activations


def char_get_avg_activations(tokens, activations):
    """Aggregates activations by averaging assuming Character-based tokenization.

    Given loaded tokens data and activations, this function aggeregates
    activations based on character-tokenized text. The activations are
    aggregated by averaging over characters.

    .. warning::
        This function is deprecated and will be removed in future versions.

    Parameters
    ----------
    tokens : dict
        Dictionary containing three lists, ``source``, ``source_aux`` and
        ``target``. Usually the output of ``data.loader.load_aux_data``.
    activations : list of numpy.ndarray
        Activations returned from ``loader.load_activations``.

    Returns
    -------
    activations : list of numpy.ndarray
        Character aggregated activations corresponding to one per actual token
        found in the untokenized text.

    """
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in tqdm(range(0, len(tokens["source_aux"]))):
        sourceIndex = 0
        thisChar = ""
        source = tokens["source"][i]
        source_aux = tokens["source_aux"][i]
        num_words = len(source)
        new_activations = np.zeros((num_words, num_neurons))

        word_boundaries = []

        for word_idx, word in enumerate(tokens["source"][i]):
            if word_idx == 0:
                word_boundaries.append(len(word) - 1)
            else:
                word_boundaries.append(len(word) + 1 + word_boundaries[-1])

        if len(word_boundaries) != num_words:
            print(i, len(word_boundaries), num_words)
        assert len(word_boundaries) == num_words
        assert (
            tokens["source_aux"][i].count("_") + 1 - tokens["source"][i].count("_")
            == num_words
        ), (
            "Number of words dont match! (line: %d, source: %d, aux: %d)\n%s\n%s"
            % (
                i + 1,
                num_words,
                tokens["source_aux"][i].count("_") + 1,
                " ".join(tokens["source"][i]),
                " ".join(tokens["source_aux"][i]),
            )
        )

        prev_idx = 0
        for word_idx, boundary in enumerate(word_boundaries):
            avg_vector = np.average(activations[i][prev_idx : boundary + 1, :], axis=0)
            new_activations[word_idx, :] = avg_vector
            prev_idx = boundary + 2

        all_activations.append(new_activations)

    return all_activations


def char_get_last_activations(tokens, activations, is_brnn=True):
    """Aggregates activations by picking the last subword assuming Character-based tokenization.

    Given loaded tokens data and activations, this function aggeregates
    activations based on character-tokenized text. The activations are
    aggregated by picking the last character for any given word.

    .. warning::
        This function is deprecated and will be removed in future versions.

    Parameters
    ----------
    tokens : dict
        Dictionary containing three lists, ``source``, ``source_aux`` and
        ``target``. Usually the output of ``data.loader.load_aux_data``.
    activations : list of numpy.ndarray
        Activations returned from ``loader.load_activations``.
    is_brnn : bool, optional
        Whether the model from which activations were extracted was bidirectional.
        Only applies for RNN models.

    Returns
    -------
    activations : list of numpy.ndarray
        Character aggregated activations corresponding to one per actual token
        found in the untokenized text.

    """
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in tqdm(range(0, len(tokens["source_aux"]))):
        sourceIndex = 0
        thisChar = ""
        source = tokens["source"][i]
        source_aux = tokens["source_aux"][i]
        num_words = len(source)
        new_activations = np.zeros((num_words, num_neurons))

        word_boundaries = []

        for word_idx, word in enumerate(tokens["source"][i]):
            if word_idx == 0:
                word_boundaries.append(len(word) - 1)
            else:
                word_boundaries.append(len(word) + 1 + word_boundaries[-1])

        if len(word_boundaries) != num_words:
            print(i, len(word_boundaries), num_words)
        assert len(word_boundaries) == num_words
        assert (
            tokens["source_aux"][i].count("_") + 1 - tokens["source"][i].count("_")
            == num_words
        ), (
            "Number of words dont match! (line: %d, source: %d, aux: %d)\n%s\n%s"
            % (
                i + 1,
                num_words,
                tokens["source_aux"][i].count("_") + 1,
                " ".join(tokens["source"][i]),
                " ".join(tokens["source_aux"][i]),
            )
        )

        rnn_boundary = int(num_neurons / 2)
        if not is_brnn:
            rnn_boundary = num_neurons

        prev_idx = 0

        for word_idx, boundary in enumerate(word_boundaries):
            # 0 - num_neurons/2: Forward
            # num_neurons/2 - : Backward
            new_activations[word_idx, :rnn_boundary] = activations[i][
                boundary, :rnn_boundary
            ]
            if is_brnn:
                new_activations[word_idx, rnn_boundary:] = activations[i][
                    prev_idx, rnn_boundary:
                ]
            prev_idx = boundary + 1

        all_activations.append(new_activations)

    return all_activations


def sent_get_last_activations(tokens, activations):
    """Gets the summary vector for the input sentences.

    Given loaded tokens data and activations, this function picks the final token's
    activations for every sentence, essentially giving summary vectors for every
    sentence in the dataset. This is mostly applicable for RNNs.

    .. note::
        Bidirectionality is currently not handled in the case of BiRNNs.

    Parameters
    ----------
    tokens : dict
        Dictionary containing three lists, ``source``, ``source_aux`` and
        ``target``. Usually the output of ``data.loader.load_aux_data``.
    activations : list of numpy.ndarray
        Activations returned from ``loader.load_activations``.

    Returns
    -------
    activations : list of numpy.ndarray
        Summary activations corresponding to one per actual sentence in the
        original text.

    """
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in tqdm(range(0, len(tokens["source"]))):
        source = tokens["source"][i]
        num_words = len(source)
        new_activations = np.zeros((1, num_neurons))

        new_activations[0, :] = activations[i][-1, :]
        all_activations.append(new_activations)

    return all_activations
