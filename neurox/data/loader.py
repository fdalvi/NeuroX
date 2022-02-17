"""Loading functions for activations, input tokens/sentences and labels

This module contains functions to load activations as well as source files with
tokens and labels. Functions that support tokenized data are also provided.
"""
import pickle
import json

import h5py
import numpy as np
import torch


def load_activations(activations_path, num_neurons_per_layer=None, is_brnn=False):
    """Load extracted activations.

    Parameters
    ----------
    activations_path : str
        Path to the activations file. Can be of type t7, pt, acts, json or hdf5
    num_neurons_per_layer : int, optional
        Number of neurons per layer - used to compute total number of layers.
        This is only necessary in the case of t7/p5/acts activations.
    is_brnn : bool, optional
        If the model used to extract activations was bidirectional (default: False)

    Returns
    -------
    activations : list of numpy.ndarray
        List of *sentence representations*, where each *sentence representation*
        is a numpy matrix of shape ``[num tokens in sentence x concatenated representation size]``
    num_layers : int
        Number of layers. This is usually representation_size/num_neurons_per_layer.
        Divide again by 2 if model was bidirectional

    """
    file_ext = activations_path.split(".")[-1]

    activations = None
    num_layers = None

    # Load activations based on type
    # Also ensure everything is on the CPU
    #   as activations may have been saved as CUDA variables
    if file_ext == "t7":
        # t7 loading requires torch < 1.0
        print("Loading seq2seq-attn activations from %s..." % (activations_path))
        assert (
            num_neurons_per_layer is not None
        ), "t7 activations require num_neurons_per_layer"
        from torch.utils.serialization import load_lua

        activations = load_lua(activations_path)["encodings"]
        activations = [a.cpu() for a in activations]
        num_layers = len(activations[0][0]) / num_neurons_per_layer
        if is_brnn:
            num_layers /= 2
    elif file_ext == "pt":
        print("Loading OpenNMT-py activations from %s..." % (activations_path))
        assert (
            num_neurons_per_layer is not None
        ), "pt activations require num_neurons_per_layer"
        activations = torch.load(activations_path)
        activations = [
            torch.stack([torch.cat(token) for token in sentence]).cpu()
            for sentence in activations
        ]
        num_layers = len(activations[0][0]) / num_neurons_per_layer
    elif file_ext == "acts":
        print("Loading generic activations from %s..." % (activations_path))
        assert (
            num_neurons_per_layer is not None
        ), "acts activations require num_neurons_per_layer"
        with open(activations_path, "rb") as activations_file:
            activations = pickle.load(activations_file)

        # Combine all layers sequentially
        print("Combining layers " + str([a[0] for a in activations]))
        activations = [a[1] for a in activations]
        num_layers = len(activations)
        num_sentences = len(activations[0])
        concatenated_activations = []
        for sentence_idx in range(num_sentences):
            sentence_acts = []
            for layer_idx in range(num_layers):
                sentence_acts.append(np.vstack(activations[layer_idx][sentence_idx]))
            concatenated_activations.append(np.concatenate(sentence_acts, axis=1))
        activations = concatenated_activations
    elif file_ext == "hdf5":
        print("Loading hdf5 activations from %s..." % (activations_path))
        representations = h5py.File(activations_path, "r")
        sentence_to_index = json.loads(representations.get("sentence_to_index")[0])
        activations = []
        # TODO: Check order
        for _, value in sentence_to_index.items():
            sentence_acts = torch.FloatTensor(representations[value])
            num_layers, sentence_length, embedding_size = (
                sentence_acts.shape[0],
                sentence_acts.shape[1],
                sentence_acts.shape[2],
            )
            num_neurons_per_layer = embedding_size
            sentence_acts = np.swapaxes(sentence_acts, 0, 1)
            sentence_acts = sentence_acts.reshape(
                sentence_length, num_layers * embedding_size
            )
            activations.append(sentence_acts.numpy())
        num_layers = len(activations[0][0]) / num_neurons_per_layer
    elif file_ext == "json":
        print("Loading json activations from %s..." % (activations_path))
        activations = []
        with open(activations_path) as fp:
            for line in fp:
                token_acts = []
                sentence_activations = json.loads(line)["features"]
                for act in sentence_activations:
                    num_neurons_per_layer = len(act["layers"][0]["values"])
                    token_acts.append(
                        np.concatenate([l["values"] for l in act["layers"]])
                    )
                activations.append(np.vstack(token_acts))

        num_layers = activations[0].shape[1] / num_neurons_per_layer
        print(len(activations), num_layers)
    else:
        assert False, "Activations must be of type t7, pt, acts, json or hdf5"

    return activations, int(num_layers)

def filter_activations_by_layers(
    train_activations, test_activations, filter_layers, rnn_size, num_layers, is_brnn
):
    """
    Filter activations so that they only contain specific layers.

    Useful for performing layer-wise analysis.

    .. warning::
        This function is deprecated and will be removed in future versions.

    Parameters
    ----------
    train_activations : list of numpy.ndarray
        List of *sentence representations* from the train set, where each
        *sentence representation* is a numpy matrix of shape
        ``[NUM_TOKENS x NUM_NEURONS]``. The method assumes that neurons from
        all layers are present, with the number of neurons in every layer given
        by ``rnn_size``
    test_activations : list of numpy.ndarray
        Similar to ``train_activations`` but with sentences from a test set.
    filter_layers : str
        A comma-separated string of the form "f1,f2,f10". "f" indicates a "forward"
        layer while "b" indicates a backword layer in a Bidirectional RNN. If the
        activations are from different kind of model, set ``is_brnn`` to ``False``
        and provide only "f" entries. The number next to "f" is the layer number,
        1-indexed. So "f1" corresponds to the embedding layer and so on.
    rnn_size : int
        Number of neurons in every layer.
    num_layers : int
        Total number of layers in the original model.
    is_brnn : bool
        Boolean indicating if the neuron activations are from a bidirectional model.

    Returns
    -------
    filtered_train_activations : list of numpy.ndarray
        Filtered train activations
    filtered_test_activations : list of numpy.ndarray
        Filtered test activations

    Notes
    -----
    For bidirectional models, the method assumes that the internal structure is
    as follows: forward layer 1 neurons, backward layer 1 neurons, forward layer
    2 neurons ...

    """
    _layers = filter_layers.split(",")

    layer_prefixes = ["f"]
    if is_brnn:
        layer_prefixes = ["f", "b"]

    # FILTER settings
    layers = list(
        range(1, num_layers + 1)
    )  # choose which layers you need the activations
    filtered_train_activations = None
    filtered_test_activations = None

    layers_idx = []
    for brnn_idx, b in enumerate(layer_prefixes):
        for l in layers:
            if "%s%d" % (b, l) in _layers:
                start_idx = brnn_idx * (num_layers * rnn_size) + (l - 1) * rnn_size
                end_idx = brnn_idx * (num_layers * rnn_size) + (l) * rnn_size

                print(
                    "Including neurons from %s%d(#%d to #%d)"
                    % (b, l, start_idx, end_idx)
                )
                layers_idx.append(np.arange(start_idx, end_idx))
    layers_idx = np.concatenate(layers_idx)

    filtered_train_activations = [a[:, layers_idx] for a in train_activations]
    filtered_test_activations = [a[:, layers_idx] for a in test_activations]

    return filtered_train_activations, filtered_test_activations


def load_aux_data(
    source_path,
    labels_path,
    source_aux_path,
    activations,
    max_sent_l,
    ignore_start_token=False,
):
    """Load word-annotated text-label pairs data represented as sentences, where
    activation extraction was performed on tokenized text. This function loads
    the source text, source tokenized text, target labels, and activations and
    tries to make them perfectly parallel, i.e. number of tokens in line N of
    source would match the number of tokens in line N of target, and number of
    tokens in source_aux will match the number of activations at index N.
    The method will delete non-matching activation/source/source_aix/target
    pairs, up to a maximum of 100 before failing. The method will also ignore
    sentences longer than the provided maximum. The activations will be modified
    in place.

    .. warning::
        This function is deprecated and will be removed in future versions.

    Parameters
    ----------
    source_path : str
        Path to the source text file, one sentence per line
    labels_path : str
        Path to the annotated labels file, one sentence per line corresponding to
        the sentences in the ``source_path`` file.
    source_aux_path : str
        Path to the source text file with tokenization, one sentence per line
    activations : list of numpy.ndarray
        Activations returned from ``loader.load_activations``
    max_sent_l : int
        Maximum length of sentences. Sentences containing more tokens will be
        ignored.
    ignore_start_token : bool, optional
        Ignore the first token. Useful if there is some line position markers
        in the source text.

    Returns
    -------
    tokens : dict
        Dictionary containing three lists, ``source``, ``source_aux`` and
        ``target``. ``source`` contains all of the sentences from``source_path``
        that were not ignored. ``source_aux`` contains all tokenized sentences
        from ``source_aux_path``. ``target`` contains the parallel set of
        annotated labels.
    """
    tokens = {"source_aux": [], "source": [], "target": []}

    skipped_lines = set()
    with open(source_aux_path) as source_aux_fp:
        for line_idx, line in enumerate(source_aux_fp):
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                print("Skipping line #%d because of length (aux)" % (line_idx))
                skipped_lines.add(line_idx)
            if ignore_start_token:
                line_tokens = line_tokens[1:]
                activations[line_idx] = activations[line_idx][1:, :]
            tokens["source_aux"].append(line_tokens)
    with open(source_path) as source_fp:
        for line_idx, line in enumerate(source_fp):
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                print("Skipping line #%d because of length (source)" % (line_idx))
                skipped_lines.add(line_idx)
            if ignore_start_token:
                line_tokens = line_tokens[1:]
            tokens["source"].append(line_tokens)

    with open(labels_path) as labels_fp:
        for line_idx, line in enumerate(labels_fp):
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                print("Skipping line #%d because of length (label)" % (line_idx))
                skipped_lines.add(line_idx)
            if ignore_start_token:
                line_tokens = line_tokens[1:]
            tokens["target"].append(line_tokens)

    assert len(tokens["source_aux"]) == len(tokens["source"]) and len(
        tokens["source_aux"]
    ) == len(tokens["target"]), (
        "Number of lines do not match (source: %d, aux: %d, target: %d)!"
        % (len(tokens["source"]), len(tokens["source_aux"]), len(tokens["target"]))
    )

    assert len(activations) == len(tokens["source"]), (
        "Number of lines do not match (activations: %d, source: %d)!"
        % (len(activations), len(tokens["source"]))
    )

    for num_deleted, line_idx in enumerate(sorted(skipped_lines)):
        print("Deleting skipped line %d" % (line_idx))
        del tokens["source_aux"][line_idx]
        del tokens["source"][line_idx]
        del tokens["target"][line_idx]

    # Check if all data is well formed (whether we have activations + labels for each
    # and every word)
    invalid_activation_idx = []
    for idx, activation in enumerate(activations):
        if activation.shape[0] == len(tokens["source_aux"][idx]) and len(
            tokens["source"][idx]
        ) == len(tokens["target"][idx]):
            pass
        else:
            invalid_activation_idx.append(idx)
            print(
                "Skipping line: ",
                idx,
                "A: %d, aux: %d, src: %d, tgt: %s"
                % (
                    activation.shape[0],
                    len(tokens["source_aux"][idx]),
                    len(tokens["source"][idx]),
                    len(tokens["target"][idx]),
                ),
            )

    assert len(invalid_activation_idx) < 100, (
        "Too many mismatches (%d) - your paths are probably incorrect or something is wrong in the data!"
        % (len(invalid_activation_idx))
    )

    for num_deleted, idx in enumerate(invalid_activation_idx):
        print(
            "Deleting line %d: %d activations, %s aux, %d source, %d target"
            % (
                idx - num_deleted,
                activations[idx - num_deleted].shape[0],
                len(tokens["source_aux"][idx - num_deleted]),
                len(tokens["source"][idx - num_deleted]),
                len(tokens["target"][idx - num_deleted]),
            )
        )
        del activations[idx - num_deleted]
        del tokens["source_aux"][idx - num_deleted]
        del tokens["source"][idx - num_deleted]
        del tokens["target"][idx - num_deleted]

    for idx, activation in enumerate(activations):
        assert activation.shape[0] == len(tokens["source_aux"][idx])
        assert len(tokens["source"][idx]) == len(tokens["target"][idx])

    return tokens


def load_data(
    source_path,
    labels_path,
    activations,
    max_sent_l,
    ignore_start_token=False,
    sentence_classification=False,
):
    """Load word-annotated text-label pairs data represented as sentences. This
    function loads the source text, target labels, and activations and tries to
    make them perfectly parallel, i.e. number of tokens in line N of source would
    match the number of tokens in line N of target, and also match the number of
    activations at index N. The method will delete non-matching activation/source/target
    pairs, up to a maximum of 100 before failing. The method will also ignore
    sentences longer than the provided maximum. The activations will be modified
    in place.

    Parameters
    ----------
    source_path : str
        Path to the source text file, one sentence per line
    labels_path : str
        Path to the annotated labels file, one sentence per line corresponding to
        the sentences in the ``source_path`` file.
    activations : list of numpy.ndarray
        Activations returned from ``loader.load_activations``
    max_sent_l : int
        Maximum length of sentences. Sentences containing more tokens will be
        ignored.
    ignore_start_token : bool, optional
        Ignore the first token. Useful if there is some line position markers
        in the source text.
    sentence_classification : bool, optional
        Flag to indicate if this is a sentence classification task, where every
        sentence actually has only a single activation (e.g. [CLS] token's
        activations in the case of BERT)

    Returns
    -------
    tokens : dict
        Dictionary containing two lists, ``source`` and ``target``. ``source``
        contains all of the sentences from ``source_path`` that were not ignored.
        ``target`` contains the parallel set of annotated labels.

    """
    tokens = {"source": [], "target": []}

    with open(source_path) as source_fp:
        for line_idx, line in enumerate(source_fp):
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                continue
            if ignore_start_token:
                line_tokens = line_tokens[1:]
                activations[line_idx] = activations[line_idx][1:, :]
            tokens["source"].append(line_tokens)

    with open(labels_path) as labels_fp:
        for line in labels_fp:
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                continue
            if ignore_start_token:
                line_tokens = line_tokens[1:]
            tokens["target"].append(line_tokens)

    assert len(tokens["source"]) == len(tokens["target"]), (
        "Number of lines do not match (source: %d, target: %d)!"
        % (len(tokens["source"]), len(tokens["target"]))
    )

    assert len(activations) == len(tokens["source"]), (
        "Number of lines do not match (activations: %d, source: %d)!"
        % (len(activations), len(tokens["source"]))
    )

    # Check if all data is well formed (whether we have activations + labels for
    # each and every word)
    invalid_activation_idx = []
    for idx, activation in enumerate(activations):
        if activation.shape[0] == len(tokens["source"][idx]) and (
            sentence_classification or activation.shape[0] == len(tokens["target"][idx])
        ):
            pass
        else:
            invalid_activation_idx.append(idx)
            print("Skipping line: ", idx)
            print(
                "A: %d, S: %d, T: %d"
                % (
                    activation.shape[0],
                    len(tokens["source"][idx]),
                    len(tokens["target"][idx]),
                )
            )

    assert len(invalid_activation_idx) < 100, (
        "Too many mismatches (%d) - your paths are probably incorrect or something is wrong in the data!"
        % (len(invalid_activation_idx))
    )

    for num_deleted, idx in enumerate(invalid_activation_idx):
        print(
            "Deleting line %d: %d activations, %d source, %d target"
            % (
                idx - num_deleted,
                activations[idx - num_deleted].shape[0],
                len(tokens["source"][idx - num_deleted]),
                len(tokens["target"][idx - num_deleted]),
            )
        )
        del activations[idx - num_deleted]
        del tokens["source"][idx - num_deleted]
        del tokens["target"][idx - num_deleted]

    for idx, activation in enumerate(activations):
        assert activation.shape[0] == len(tokens["source"][idx])
        if not sentence_classification:
            assert activation.shape[0] == len(tokens["target"][idx])

    # TODO: Return activations
    return tokens


def load_sentence_data(source_path, labels_path, activations):
    """Loads sentence-annotated text-label pairs. This function loads the source
    text, target labels, and activations and tries to make them perfectly
    parallel, i.e. number of tokens in line N of source would
    match the number of activations at index N. The method will delete
    non-matching activation/source pairs. The activations will be modified
    in place.

    Parameters
    ----------
    source_path : str
        Path to the source text file, one sentence per line
    labels_path : str
        Path to the annotated labels file, one sentence per line corresponding to
        the sentences in the ``source_path`` file.
    activations : list of numpy.ndarray
        Activations returned from ``loader.load_activations``

    Returns
    -------
    tokens : dict
        Dictionary containing two lists, ``source`` and ``target``. ``source``
        contains all of the sentences from ``source_path`` that were not ignored.
        ``target`` contains the parallel set of annotated labels.

    """
    tokens = {"source": [], "target": []}

    with open(source_path) as source_fp:
        for line_idx, line in enumerate(source_fp):
            tokens["source"].append(["sentence_%d" % (line_idx)])

    with open(labels_path) as labels_fp:
        for line in labels_fp:
            line_tokens = line.strip().split()
            tokens["target"].append(line_tokens)

    assert len(tokens["source"]) == len(tokens["target"]), (
        "Number of lines do not match (source: %d, target: %d)!"
        % (len(tokens["source"]), len(tokens["target"]))
    )

    assert len(activations) == len(tokens["source"]), (
        "Number of lines do not match (activations: %d, source: %d)!"
        % (len(activations), len(tokens["source"]))
    )

    # Check if all data is well formed (whether we have activations + labels for
    # each and every word)
    for idx, activation in enumerate(activations):
        assert activation.shape[0] == len(tokens["source"][idx])

    return tokens
