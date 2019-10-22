import pickle
import json

import h5py
import numpy as np
import torch

from torch.utils.serialization import load_lua


def load_activations(activations_path, num_neurons_per_layer, is_brnn=True):
    """Load extracted activations.

    Arguments:
    activations_path (str): Path to the activations file. Can be of type t7, pt, acts
    num_neurons_per_layer (int): Number of neurons per layer - used to compute total
        number of layers.
    is_brnn (bool): If the model used to extract activations was bidirectional (default: True)

    Returns:
    activations (list x numpy matrix): List of `sentence representations`, where each
        `sentence representation` is a numpy matrix of shape
        (num tokens in sentence x concatenated representation size)
    num_layers (int): Number of layers. This is usually representation_size/num_neurons_per_layer.
        Divide again by 2 if model was bidirectional
    """
    file_ext = activations_path.split(".")[-1]

    activations = None
    num_layers = None

    # Load activations based on type
    # Also ensure everything is on the CPU
    #   as activations may have been saved as CUDA variables
    if file_ext == "t7":
        print("Loading seq2seq-attn activations from %s..." % (activations_path))
        activations = load_lua(activations_path)["encodings"]
        activations = [a.cpu() for a in activations]
        num_layers = len(activations[0][0]) / num_neurons_per_layer
        if is_brnn:
            num_layers /= 2
    elif file_ext == "pt":
        print("Loading OpenNMT-py activations from %s..." % (activations_path))
        activations = torch.load(activations_path)
        activations = [
            torch.stack([torch.cat(token) for token in sentence]).cpu()
            for sentence in activations
        ]
        num_layers = len(activations[0][0]) / num_neurons_per_layer
    elif file_ext == "acts":
        print("Loading generic activations from %s..." % (activations_path))
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
        for _, value in sentence_to_index.items():
            sentence_acts = torch.FloatTensor(representations[value])
            num_layers, sentence_length, embedding_size = (
                sentence_acts.shape[0],
                sentence_acts.shape[1],
                sentence_acts.shape[2],
            )
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
                sentence_activations = json.loads(line)['features']
                for act in sentence_activations:
                    token_acts.append(np.concatenate([l['values'] for l in act['layers']]))
                activations.append(np.vstack(token_acts))

        num_layers = activations[0].shape[1] / num_neurons_per_layer
        print(len(activations), num_layers)
    else:
        assert False, "Activations must be of type t7, pt, acts or hdf5"

    return activations, int(num_layers)


def load_aux_data(
    source_path,
    labels_path,
    source_aux_path,
    activations,
    max_sent_l,
    ignore_start_token=False,
):
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

    assert len(invalid_activation_idx) < 100, \
        "Too many mismatches (%d) - your paths are probably incorrect or something is wrong in the data!" % (len(invalid_activation_idx))

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

    assert len(invalid_activation_idx) < 100, \
        "Too many mismatches (%d) - your paths are probably incorrect or something is wrong in the data!" % (len(invalid_activation_idx))

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

    return tokens

def load_sentence_data(
    source_path,
    labels_path,
    activations
):
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
    invalid_activation_idx = []
    for idx, activation in enumerate(activations):
        assert activation.shape[0] == len(tokens["source"][idx])

    return tokens
